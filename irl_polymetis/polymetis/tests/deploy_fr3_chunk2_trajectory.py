import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torchcontrol as toco
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from polymetis import GripperInterface, RobotInterface


CAM_MAPPING = {
    "348122070707": "observation.images.left",
    "352122270841": "observation.images.wrist",
    "347622075736": "observation.images.right",
}

TASK_PROMPT = "pick up the carrot and put it in the yellow box"

CONTROL_HZ = 10
REPLAN_INTERVAL = 16

SHOW_CAMERA_PREVIEW = True
PREVIEW_WINDOW_NAME = "RealSense Multi-Camera Preview (Left | Wrist | Right)"

# This deployer does not use start_joint_impedance/update_desired_joint_positions.
# It executes each predicted chunk as a short joint-space trajectory.
GAIN_SCALE = 1.0
MAX_JOINT_STEP = 0.25
RESET_TIME = 3.0
POLICY_TIMEOUT_MARGIN = 3.0

# Dataset convention after restore_quat_action2joints.py:
# action[7] = 1 means gripper open, action[7] = 0 means gripper close.
GRIPPER_ONE_MEANS_OPEN = True
GRIPPER_OPEN_WIDTH = 0.07


def _as_tensor(values):
    return torch.tensor(values, dtype=torch.float32)


def _limit_joint_steps(targets, start_qpos, max_joint_step):
    limited_targets = []
    previous = np.asarray(start_qpos, dtype=np.float32)
    for target in targets:
        target = np.asarray(target, dtype=np.float32)
        delta = target - previous
        max_abs = float(np.max(np.abs(delta)))
        if max_joint_step is not None and max_abs > max_joint_step:
            target = previous + delta / max_abs * max_joint_step
        limited_targets.append(target.astype(np.float32))
        previous = target
    return np.asarray(limited_targets, dtype=np.float32)


def _build_joint_trajectory(start_qpos, targets, control_hz, robot_hz):
    start_qpos = np.asarray(start_qpos, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    dt = 1.0 / float(control_hz)
    steps_per_action = max(1, int(round(float(robot_hz) * dt)))

    pos_traj = []
    vel_traj = []
    previous = start_qpos
    for target in targets:
        target = np.asarray(target, dtype=np.float32)
        velocity = (target - previous) / dt
        for step in range(steps_per_action):
            alpha = step / float(steps_per_action)
            waypoint = (1.0 - alpha) * previous + alpha * target
            pos_traj.append(_as_tensor(waypoint))
            vel_traj.append(_as_tensor(velocity))
        previous = target

    pos_traj.append(_as_tensor(targets[-1]))
    vel_traj.append(torch.zeros(7, dtype=torch.float32))
    return pos_traj, vel_traj


class FrankaPi0TrajectoryDeployer:
    def __init__(self):
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host="localhost",
            port=8000,
        )

        self.robot = RobotInterface(ip_address="localhost")
        self.gripper = GripperInterface(ip_address="localhost")

        self.pipelines = {}
        self._last_gripper_open = None
        self._gripper_lock = threading.Lock()

        if SHOW_CAMERA_PREVIEW:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW_NAME, 1440, 360)
        self._init_cameras()

    def _init_cameras(self):
        for serial in CAM_MAPPING.keys():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            self.pipelines[serial] = pipeline
        print("All cameras started.")

    def _terminate_policy_quiet(self):
        try:
            self.robot.terminate_current_policy(return_log=False)
        except Exception:
            pass

    def _update_camera_preview(self, previews):
        display_order = [
            "observation.images.left",
            "observation.images.wrist",
            "observation.images.right",
        ]
        ordered_previews = [previews[key] for key in display_order if key in previews]
        if len(ordered_previews) != 3:
            return

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
            self._update_camera_preview(previews)

        qpos = self.robot.get_joint_positions().numpy()
        gripper_width = self.gripper.get_state().width
        gripper_binary = 1.0 if gripper_width > 0.05 else 0.0
        obs["observation.state"] = np.append(qpos, gripper_binary).astype(np.float32)
        return obs

    def _set_gripper_from_action(self, gripper_action):
        should_open = bool(float(gripper_action) > 0.5)
        if not GRIPPER_ONE_MEANS_OPEN:
            should_open = not should_open

        with self._gripper_lock:
            if should_open == self._last_gripper_open:
                return
            self._last_gripper_open = should_open

        if should_open:
            self.gripper.goto(
                width=GRIPPER_OPEN_WIDTH,
                speed=0.1,
                force=20.0,
                blocking=False,
            )
        else:
            self.gripper.grasp(
                grasp_width=0.01,
                speed=0.1,
                force=30.0,
                epsilon_inner=0.01,
                epsilon_outer=0.06,
                blocking=False,
            )

    def _start_gripper_thread(self, gripper_actions, chunk_dt):
        stop_event = threading.Event()

        def worker():
            for gripper_action in gripper_actions:
                if stop_event.is_set():
                    return
                self._set_gripper_from_action(gripper_action)
                if stop_event.wait(chunk_dt):
                    return

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread, stop_event

    def _execute_action_chunk(self, action_chunk):
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.shape[1] < 8:
            raise ValueError(
                "Expected actions with shape [N, 8] = [q1..q7, gripper], "
                f"got {action_chunk.shape}"
            )

        chunk = action_chunk[: min(REPLAN_INTERVAL, len(action_chunk))]
        current_qpos = self.robot.get_joint_positions().numpy().astype(np.float32)
        raw_targets = chunk[:, :7]
        targets = _limit_joint_steps(raw_targets, current_qpos, MAX_JOINT_STEP)

        first_delta = targets[0] - current_qpos
        print("current:", current_qpos)
        print("target :", targets[0])
        print("delta  :", first_delta, "max_abs:", np.max(np.abs(first_delta)))

        robot_hz = float(getattr(self.robot, "hz", self.robot.metadata.hz))
        pos_traj, vel_traj = _build_joint_trajectory(
            current_qpos,
            targets,
            CONTROL_HZ,
            robot_hz,
        )

        torch_policy = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=pos_traj,
            joint_vel_trajectory=vel_traj,
            Kq=self.robot.Kq_default * GAIN_SCALE,
            Kqd=self.robot.Kqd_default * GAIN_SCALE,
            Kx=self.robot.Kx_default,
            Kxd=self.robot.Kxd_default,
            robot_model=self.robot.robot_model,
            ignore_gravity=self.robot.use_grav_comp,
        )

        chunk_dt = 1.0 / float(CONTROL_HZ)
        gripper_thread, gripper_stop = self._start_gripper_thread(chunk[:, 7], chunk_dt)

        self._terminate_policy_quiet()
        duration = len(chunk) * chunk_dt
        start_t = time.time()
        try:
            self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)
            while self.robot.is_running_policy():
                if time.time() - start_t > duration + POLICY_TIMEOUT_MARGIN:
                    raise TimeoutError("Joint trajectory policy timed out.")
                time.sleep(0.02)
        finally:
            gripper_stop.set()
            gripper_thread.join(timeout=1.0)

    def close(self):
        self._terminate_policy_quiet()
        for pipeline in self.pipelines.values():
            pipeline.stop()
        if SHOW_CAMERA_PREVIEW:
            cv2.destroyWindow(PREVIEW_WINDOW_NAME)

    def run(self):
        print(f"Starting task: {TASK_PROMPT}")
        print("Using chunked joint trajectory execution; no persistent joint impedance loop.")
        try:
            while True:
                obs = self.get_observation()
                result = self.client.infer(obs)
                if "actions" not in result:
                    raise KeyError(f"Policy response missing 'actions': {result.keys()}")
                self._execute_action_chunk(result["actions"])
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            self.close()


if __name__ == "__main__":
    deployer = FrankaPi0TrajectoryDeployer()
    deployer.run()
