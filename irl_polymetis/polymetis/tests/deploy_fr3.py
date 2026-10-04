import time
import numpy as np
import pyrealsense2 as rs
import cv2
from polymetis import RobotInterface, GripperInterface
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import torch
# ================= 配置区 =================
# 1. 相机序列号映射 (务必与你训练时的视角一致)
CAM_MAPPING = {
    '348122070707': 'observation.images.left',
    '352122270841': 'observation.images.wrist',
    '347622075736': 'observation.images.right'
}
# 2. 任务指令
TASK_PROMPT = "Pick up the corn and put it in the yellow box"

CONTROL_HZ = 10 
SHOW_CAMERA_PREVIEW = True
PREVIEW_CAMERA_SERIAL = "352122270841"
PREVIEW_WINDOW_NAME = f"RealSense Preview - {PREVIEW_CAMERA_SERIAL}"
# ==========================================

class FrankaPi0Deployer:
    def __init__(self):
        # 初始化模型客户端
        self.client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)
        
        # 初始化 Franka
        self.robot = RobotInterface(ip_address="localhost")
        self.gripper = GripperInterface(ip_address="localhost")
        
        # 初始化相机管线
        self.pipelines = {}
        self._last_gripper_open = None
        if SHOW_CAMERA_PREVIEW:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW_NAME, 960, 540)
        self._init_cameras()

    def _init_cameras(self):
        for serial in CAM_MAPPING.keys():
            p = rs.pipeline()
            conf = rs.config()
            conf.enable_device(serial)
            conf.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            p.start(conf)
            self.pipelines[serial] = p
        print("✅ 所有相机已启动")

    def get_observation(self):
        obs = {"task": TASK_PROMPT}
        
        # 1. 获取并处理图像
        for serial, key in CAM_MAPPING.items():
            frames = self.pipelines[serial].wait_for_frames()
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())

            if SHOW_CAMERA_PREVIEW and serial == PREVIEW_CAMERA_SERIAL:
                preview = img.copy()
                cv2.putText(
                    preview,
                    f"{serial} | press q / Esc to stop",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(PREVIEW_WINDOW_NAME, preview)
                key_code = cv2.waitKey(1) & 0xFF
                if key_code in (27, ord("q")):
                    raise KeyboardInterrupt

            # 必须进行 RGB 转换和 224 缩放，匹配训练分布
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs[key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_rgb, 224, 224)
            )

        # 2. 获取机器人状态 (7关节 + 1夹爪状态)
        qpos = self.robot.get_joint_positions().numpy()
        gripper_state = self.gripper.get_state()
        # 参考你代码里的逻辑：根据宽度转为 0/1 状态
        #gripper_state = self.robot.get_gripper_state() # 需根据你实际的读取接口
        gripper_binary = 1.0 if gripper_state.width > 0.05 else 0.0
        
        obs["observation.state"] = np.append(qpos, gripper_binary).astype(np.float32)
        return obs

    def close(self):
        for p in self.pipelines.values():
            p.stop()
        if SHOW_CAMERA_PREVIEW:
            cv2.destroyWindow(PREVIEW_WINDOW_NAME)
        try:
            self.robot.terminate_current_policy()
        except Exception:
            pass

    def _start_joint_controller(self):
        current_qpos = self.robot.get_joint_positions()
        self.robot.start_joint_impedance(
            Kq=self.robot.Kq_default,
            Kqd=self.robot.Kqd_default,
            adaptive=False,
        )
        self.robot.update_desired_joint_positions(current_qpos)

    def run(self):
        print(f"🚀 开始执行任务: {TASK_PROMPT}")
        try:
            # Current training setup predicts future joint states, so deployment should
            # command joint targets instead of end-effector poses.
            self._start_joint_controller()
            while True:
                start_t = time.time()
                
                # 1. 采集当前观察
                obs = self.get_observation()
                
                # 2. 远程推理请求
                # 返回 action_chunk 形状为 (10, 8)
                result = self.client.infer(obs)
                
                action = result["actions"][1]  # 8维：[q1..q7, gripper]
                #print(action)
                current_qpos = self.robot.get_joint_positions().numpy()
                target_qpos_np = action[:7]
                dq = target_qpos_np - current_qpos
                print("current:", current_qpos)
                print("target :", target_qpos_np)
                print("delta  :", dq, "max_abs:", np.max(np.abs(dq)))

                target_qpos = torch.tensor(action[:7], dtype=torch.float32)
                target_gripper = action[7]

                # Policy output is already mapped back to absolute joint-space targets.
                self.robot.update_desired_joint_positions(target_qpos)
                # 5. 夹爪控制逻辑
                should_open = bool(target_gripper > 0.5)
                if should_open != self._last_gripper_open:
                    if should_open:
                        self.gripper.goto(width=0.08, speed=0.1, force=20) # 执行开启动作
                    else:
                        self.gripper.grasp(speed=0.1, force=20) # 执行关闭动作
                    self._last_gripper_open = should_open

                # 维持 10Hz
                sleep_time = (1.0 / CONTROL_HZ) - (time.time() - start_t)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("🛑 停止运行")
        finally:
            self.close()

if __name__ == "__main__":
    deployer = FrankaPi0Deployer()
    deployer.run()
