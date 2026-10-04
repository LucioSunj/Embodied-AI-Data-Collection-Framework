import time

import cv2
import grpc
import numpy as np
import pyrealsense2 as rs
import torch
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from polymetis import GripperInterface, RobotInterface
from scipy.spatial.transform import Rotation as R


CAM_MAPPING = {
    "348122070707": "observation.images.left",
    "352122270841": "observation.images.wrist",
    "347622075736": "observation.images.right",
}

TASK_PROMPT = "pick up all things and put them in the yellow box"

POLICY_HOST = "localhost"
POLICY_PORT = 8000
ROBOT_IP = "localhost"

CONTROL_HZ = 10
REPLAN_INTERVAL = 16

# The pi05_fr3_gello_lerobot_multiple_tasks config is intended to return:
# [x, y, z, roll, pitch, yaw, gripper].
ACTION_LAYOUT = "xyz_rpy_gripper"
# Optional values: "xyz_rpy_gripper", "xyz_quat_gripper", "auto".
RPY_ORDER = "xyz"

# Keep these limits conservative for first hardware tests.
CLIP_TO_WORKSPACE = True
WORKSPACE_LOW = np.array([0.25, -0.45, 0.08], dtype=np.float32)
WORKSPACE_HIGH = np.array([0.80, 0.45, 0.65], dtype=np.float32)
MAX_CARTESIAN_STEP = 0.03
MAX_ROTATION_STEP = 0.12

# Matches deploy_fr3_chunk2.py behavior: gripper value > 0.5 means close.
# If your policy outputs observation-style gripper state where 1=open, set this to False.
GRIPPER_ONE_MEANS_CLOSE = True
GRIPPER_OPEN_WIDTH = 0.07
GRIPPER_SPEED = 0.1
GRIPPER_OPEN_FORCE = 20.0
GRIPPER_CLOSE_FORCE = 30.0

SHOW_CAMERA_PREVIEW = True
PREVIEW_WINDOW_NAME = "RealSense Multi-Camera Preview (Left | Wrist | Right)"


class FrankaPi0CartesianDeployer:
    def __init__(self):
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=POLICY_HOST,
            port=POLICY_PORT,
        )

        self.robot = RobotInterface(ip_address=ROBOT_IP)
        self.gripper = GripperInterface(ip_address=ROBOT_IP)

        self.pipelines = {}
        self._last_gripper_closed = None

        self.action_chunk = None
        self.chunk_idx = 0

        if SHOW_CAMERA_PREVIEW:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW_NAME, 1440, 360)

        self._init_cameras()

    def _init_cameras(self):
        ctx = rs.context()
        connected = {}
        for device in ctx.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            connected[serial] = name

        missing_serials = [serial for serial in CAM_MAPPING if serial not in connected]
        if missing_serials:
            connected_text = ", ".join(
                f"{serial} ({name})" for serial, name in sorted(connected.items())
            )
            missing_text = ", ".join(missing_serials)
            raise RuntimeError(
                "Missing RealSense camera(s): "
                f"{missing_text}. Connected camera(s): {connected_text or 'none'}."
            )

        for serial in CAM_MAPPING:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            self.pipelines[serial] = pipeline
        print("All cameras started.")

    def get_observation(self):
        obs = {"task": TASK_PROMPT}
        previews = {}

        for serial, key in CAM_MAPPING.items():
            frames = self.pipelines[serial].wait_for_frames()
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())

            if SHOW_CAMERA_PREVIEW:
                preview = img.copy()
                view_label = key.split(".")[-1].upper()
                cv2.putText(
                    preview,
                    f"{view_label} ({serial})",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                previews[key] = preview

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs[key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_rgb, 224, 224)
            )

        if SHOW_CAMERA_PREVIEW:
            display_order = (
                "observation.images.left",
                "observation.images.wrist",
                "observation.images.right",
            )
            ordered_previews = [previews[k] for k in display_order if k in previews]

            if len(ordered_previews) == 3:
                combined_img = np.hstack(ordered_previews)
                cv2.putText(
                    combined_img,
                    "Press 'q' or 'ESC' to stop",
                    (15, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(PREVIEW_WINDOW_NAME, combined_img)
                key_code = cv2.waitKey(1) & 0xFF
                if key_code in (27, ord("q")):
                    raise KeyboardInterrupt

        qpos = self.robot.get_joint_positions().numpy()
        gripper_state = self.gripper.get_state()
        gripper_binary = 1.0 if gripper_state.width > 0.05 else 0.0

        obs["observation.state"] = np.append(qpos, gripper_binary).astype(np.float32)
        return obs

    def close(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()

        if SHOW_CAMERA_PREVIEW:
            cv2.destroyWindow(PREVIEW_WINDOW_NAME)

        try:
            self.robot.terminate_current_policy()
        except Exception:
            pass

    def _start_cartesian_controller(self):
        current_pos, current_quat = self.robot.get_ee_pose()
        self.robot.start_cartesian_impedance(
            Kx=self.robot.Kx_default,
            Kxd=self.robot.Kxd_default,
        )
        self.robot.update_desired_ee_pose(
            position=current_pos,
            orientation=current_quat,
        )
        print("Cartesian impedance controller started.")

    def _should_replan(self):
        if self.action_chunk is None:
            return True

        chunk_len = len(self.action_chunk)
        if self.chunk_idx >= chunk_len:
            return True

        if self.chunk_idx >= min(REPLAN_INTERVAL, chunk_len):
            return True

        return False

    def _sanitize_position(self, target_pos, current_pos):
        target_pos = np.asarray(target_pos, dtype=np.float32).copy()

        if CLIP_TO_WORKSPACE:
            target_pos = np.clip(target_pos, WORKSPACE_LOW, WORKSPACE_HIGH)

        if MAX_CARTESIAN_STEP is not None:
            delta = target_pos - current_pos
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > MAX_CARTESIAN_STEP:
                target_pos = current_pos + delta / delta_norm * MAX_CARTESIAN_STEP

        return target_pos.astype(np.float32)

    def _parse_cartesian_action(self, action, current_pos):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if ACTION_LAYOUT not in {"xyz_rpy_gripper", "xyz_quat_gripper", "auto"}:
            raise ValueError(f"Unknown ACTION_LAYOUT: {ACTION_LAYOUT}")

        use_quat_layout = ACTION_LAYOUT == "xyz_quat_gripper"
        if ACTION_LAYOUT == "auto" and action.shape[0] >= 8:
            quat_norm = float(np.linalg.norm(action[3:7]))
            use_quat_layout = 0.8 <= quat_norm <= 1.2

        min_dim = 8 if use_quat_layout else 7
        if action.shape[0] < min_dim:
            raise ValueError(
                "Expected Cartesian action with enough dimensions, "
                f"got shape {action.shape}"
            )

        target_pos = self._sanitize_position(action[:3], current_pos)

        if use_quat_layout:
            target_quat = action[3:7].astype(np.float32)
            target_quat = target_quat / (np.linalg.norm(target_quat) + 1e-8)
            target_rpy = R.from_quat(target_quat).as_euler(RPY_ORDER).astype(np.float32)
            target_gripper = float(action[7])
        else:
            target_rpy = ((action[3:6] + np.pi) % (2.0 * np.pi)) - np.pi
            target_quat = R.from_euler(RPY_ORDER, target_rpy).as_quat().astype(np.float32)
            target_gripper = float(action[6])

        return target_pos, target_quat, target_rpy.astype(np.float32), target_gripper

    def _limit_orientation_step(self, target_quat, current_quat):
        if MAX_ROTATION_STEP is None:
            return target_quat.astype(np.float32)

        current_rot = R.from_quat(current_quat)
        target_rot = R.from_quat(target_quat)
        relative_rot = target_rot * current_rot.inv()
        angle = float(relative_rot.magnitude())

        if angle <= MAX_ROTATION_STEP:
            return target_quat.astype(np.float32)

        limited_rot = R.from_rotvec(
            relative_rot.as_rotvec() * (MAX_ROTATION_STEP / angle)
        ) * current_rot
        return limited_rot.as_quat().astype(np.float32)

    def _command_gripper(self, target_gripper):
        gripper_is_one = bool(target_gripper > 0.5)
        should_close = gripper_is_one if GRIPPER_ONE_MEANS_CLOSE else not gripper_is_one

        if should_close == self._last_gripper_closed:
            return

        if should_close:
            self.gripper.grasp(
                grasp_width=0.01,
                speed=GRIPPER_SPEED,
                force=GRIPPER_CLOSE_FORCE,
                epsilon_inner=0.01,
                epsilon_outer=0.06,
                blocking=True,
            )
        else:
            self.gripper.goto(
                width=GRIPPER_OPEN_WIDTH,
                speed=GRIPPER_SPEED,
                force=GRIPPER_OPEN_FORCE,
                blocking=False,
            )

        self._last_gripper_closed = should_close

    def _recover_cartesian_controller(self):
        print("Controller RPC error. Restarting Cartesian impedance controller.")
        self.action_chunk = None
        self.chunk_idx = 0
        time.sleep(0.2)
        self._start_cartesian_controller()

    def run(self):
        print(f"Starting task: {TASK_PROMPT}")

        try:
            self._start_cartesian_controller()

            while True:
                start_t = time.time()
                obs = self.get_observation()

                if self._should_replan():
                    result = self.client.infer(obs)
                    if "actions" not in result:
                        raise KeyError(f"Policy response missing 'actions': {result.keys()}")

                    self.action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    if self.action_chunk.ndim == 1:
                        self.action_chunk = self.action_chunk[None, :]
                    self.chunk_idx = 0

                action = self.action_chunk[self.chunk_idx]
                self.chunk_idx += 1

                current_pos, current_quat = self.robot.get_ee_pose()
                current_pos_np = current_pos.numpy().astype(np.float32)
                current_rpy = R.from_quat(current_quat.numpy()).as_euler(RPY_ORDER)

                target_pos_np, target_quat_np, target_rpy_np, target_gripper = (
                    self._parse_cartesian_action(action, current_pos_np)
                )
                target_quat_np = self._limit_orientation_step(
                    target_quat_np,
                    current_quat.numpy(),
                )
                target_rpy_limited = R.from_quat(target_quat_np).as_euler(RPY_ORDER)

                if self.chunk_idx == 1:
                    print("current xyz:", current_pos_np)
                    print("target  xyz:", target_pos_np)
                    print("delta   xyz:", target_pos_np - current_pos_np)
                    print("current rpy:", current_rpy)
                    print("raw target rpy:", target_rpy_np)
                    print("cmd target rpy:", target_rpy_limited)
                    print("gripper:", target_gripper)

                try:
                    update_idx = self.robot.update_desired_ee_pose(
                        position=torch.tensor(target_pos_np, dtype=torch.float32),
                        orientation=torch.tensor(target_quat_np, dtype=torch.float32),
                    )
                    if update_idx < 0:
                        print("IK failed for target pose; forcing a replan.")
                        self.action_chunk = None
                        self.chunk_idx = 0
                        continue
                except grpc.RpcError:
                    self._recover_cartesian_controller()
                    continue

                self._command_gripper(target_gripper)

                sleep_time = (1.0 / CONTROL_HZ) - (time.time() - start_t)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            self.close()


if __name__ == "__main__":
    deployer = FrankaPi0CartesianDeployer()
    deployer.run()
