import time

import cv2
import grpc
import numpy as np
import pyrealsense2 as rs
import torch
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from polymetis import GripperInterface, RobotInterface


CAM_MAPPING = {
    "348122070707": "observation.images.left",
    "347622075736": "observation.images.right",
}

TASK_PROMPT = "pick up all things and put them in the yellow box"
CONTROL_HZ = 10
REPLAN_INTERVAL = 16
GRIPPER_COMMAND_SETTLE_SEC = 0.08

SHOW_CAMERA_PREVIEW = True
PREVIEW_WINDOW_NAME = "RealSense Multi-Camera Preview (Left | Right)"
FILL_MISSING_WRIST_IMAGE = True


class FrankaPi0NoWristDeployer:
    def __init__(self):
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host="localhost",
            port=8000,
        )

        self.robot = RobotInterface(ip_address="localhost")
        self.gripper = GripperInterface(ip_address="localhost")

        self.pipelines = {}
        self._last_gripper_open = None

        self.action_chunk = None
        self.chunk_idx = 0
        self.controller_restart_count = 0

        if SHOW_CAMERA_PREVIEW:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW_NAME, 960, 360)

        self._init_cameras()

    def _init_cameras(self):
        for serial in CAM_MAPPING:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            self.pipelines[serial] = pipeline
        print("✅ 左/右相机已启动，未启用腕部相机")

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

        if FILL_MISSING_WRIST_IMAGE:
            obs["observation.images.wrist"] = np.zeros((224, 224, 3), dtype=np.uint8)

        if SHOW_CAMERA_PREVIEW:
            display_order = [
                "observation.images.left",
                "observation.images.right",
            ]
            ordered_previews = [previews[k] for k in display_order if k in previews]

            if len(ordered_previews) == len(display_order):
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
            self.robot.terminate_current_policy(return_log=False)
        except Exception:
            pass

    def _terminate_controller_quiet(self):
        try:
            self.robot.terminate_current_policy(return_log=False)
        except Exception:
            pass

    def _start_joint_controller(self):
        self._terminate_controller_quiet()
        current_qpos = self.robot.get_joint_positions()

        default_kq = self.robot.Kq_default
        default_kqd = self.robot.Kqd_default

        stiff_factor = 2
        rigid_kq = default_kq * stiff_factor
        damp_factor = stiff_factor**0.5
        rigid_kqd = default_kqd * damp_factor

        print("\n" + "=" * 50)
        print(f"⚠️ 注意: 已启用【高刚度】阻抗控制！刚度放大了 {stiff_factor} 倍。")
        print("=" * 50 + "\n")

        self.robot.start_joint_impedance(
            Kq=rigid_kq,
            Kqd=rigid_kqd,
            adaptive=False,
        )
        time.sleep(0.1)
        self.robot.update_desired_joint_positions(current_qpos)
        print("🛡️ 高刚度关节阻抗控制器已启动")

    def _ensure_joint_controller(self):
        try:
            if self.robot.is_running_policy():
                return
        except grpc.RpcError:
            pass

        self.controller_restart_count += 1
        print(
            "⚠️ 关节阻抗控制器未运行，正在重启 "
            f"(第 {self.controller_restart_count} 次)..."
        )
        self._start_joint_controller()

    def _safe_update_desired_joint_positions(self, target_qpos):
        self._ensure_joint_controller()
        try:
            return self.robot.update_desired_joint_positions(target_qpos)
        except grpc.RpcError as exc:
            self.controller_restart_count += 1
            print(
                "⚠️ 控制器 update 失败，正在重启后重发目标 "
                f"(第 {self.controller_restart_count} 次): {exc}"
            )
            self._start_joint_controller()
            time.sleep(0.05)
            return self.robot.update_desired_joint_positions(target_qpos)

    def _stop_gripper_before_command(self):
        command_queue = getattr(self.gripper, "_command_queue", None)
        if command_queue is not None:
            command_queue.join()

        stop = getattr(self.gripper, "stop", None)
        if stop is None:
            return

        try:
            stop()
            time.sleep(GRIPPER_COMMAND_SETTLE_SEC)
        except Exception as exc:
            print(f"⚠️ 夹爪 stop() 失败，继续发送下一条命令: {exc}")

    def _safe_gripper_grasp(self):
        self._stop_gripper_before_command()
        self.gripper.grasp(
            grasp_width=0.02,
            speed=0.1,
            force=30.0,
            epsilon_inner=0.01,
            epsilon_outer=0.1,
            blocking=True,
        )

    def _safe_gripper_open(self):
        self._stop_gripper_before_command()
        self.gripper.goto(
            width=0.07,
            speed=0.1,
            force=20.0,
            blocking=True,
        )

    def _should_replan(self):
        if self.action_chunk is None:
            return True

        chunk_len = len(self.action_chunk)
        if self.chunk_idx >= chunk_len:
            return True

        if self.chunk_idx >= min(REPLAN_INTERVAL, chunk_len):
            return True

        return False

    def run(self):
        print(f"🚀 开始执行任务: {TASK_PROMPT}")
        try:
            self._start_joint_controller()
            while True:
                start_t = time.time()

                obs = self.get_observation()

                if self._should_replan():
                    result = self.client.infer(obs)
                    if "actions" not in result:
                        raise KeyError(f"服务端响应缺少 'actions': {result.keys()}")

                    self.action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    self.chunk_idx = 0

                action = self.action_chunk[self.chunk_idx]
                self.chunk_idx += 1

                current_qpos = self.robot.get_joint_positions().numpy()
                target_qpos_np = action[:7]
                delta_qpos = target_qpos_np - current_qpos

                if self.chunk_idx == 1:
                    print("current:", current_qpos)
                    print("target :", target_qpos_np)
                    print("delta  :", delta_qpos, "max_abs:", np.max(np.abs(delta_qpos)))

                target_qpos = torch.tensor(target_qpos_np, dtype=torch.float32)
                target_gripper = action[7]

                self._safe_update_desired_joint_positions(target_qpos)

                should_open = bool(target_gripper > 0.5)
                if should_open != self._last_gripper_open:
                    if should_open:
                        self._safe_gripper_grasp()
                    else:
                        self._safe_gripper_open()

                    self._last_gripper_open = should_open

                sleep_time = (1.0 / CONTROL_HZ) - (time.time() - start_t)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("🛑 停止运行")
        finally:
            self.close()


if __name__ == "__main__":
    deployer = FrankaPi0NoWristDeployer()
    deployer.run()
