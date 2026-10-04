import time
import numpy as np
import pyrealsense2 as rs
import cv2
from polymetis import RobotInterface, GripperInterface
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import torch

CAM_MAPPING = {
    '348122070707': 'observation.images.left',
    '352122270841': 'observation.images.wrist',
    '347622075736': 'observation.images.right'
}
# 2. 任务指令
TASK_PROMPT = "Pick up the carrot and put it in the yellow box"

CONTROL_HZ = 10 


REPLAN_INTERVAL = 16

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
        
        
        self.action_chunk = None
        self.chunk_idx = 0
        
        if SHOW_CAMERA_PREVIEW:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW_NAME, 640, 480)
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


            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs[key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_rgb, 224, 224)
            )

        qpos = self.robot.get_joint_positions().numpy()
        gripper_state = self.gripper.get_state()
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

    # ✨ 新增：判断是否需要请求模型重新规划
    def _should_replan(self):
        if self.action_chunk is None:
            return True

        chunk_len = len(self.action_chunk)
        if self.chunk_idx >= chunk_len:
            return True

        # 如果已经执行了设定的步数，或者超过了 chunk 的剩余长度，则重新规划
        if self.chunk_idx >= min(REPLAN_INTERVAL, chunk_len):
            return True

        return False

    def run(self):
        print(f"🚀 开始执行任务: {TASK_PROMPT}")
        try:
            self._start_joint_controller()
            while True:
                start_t = time.time()
                
                # 1. 采集当前观察 
                # (注意：每次循环都必须调用，以清理相机缓存，保持画面实时)
                obs = self.get_observation()
                
                # ✨ 2. 判断是否触发推理
                if self._should_replan():
                    result = self.client.infer(obs)
                    if "actions" not in result:
                        raise KeyError(f"服务端响应缺少 'actions': {result.keys()}")
                    
                    self.action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    self.chunk_idx = 0
                    #print(f"🔄 Replan Triggered! 拿到新预测，开始执行第 0 步...")

                # ✨ 3. 从现有的 Chunk 中提取当前步的动作
                action = self.action_chunk[self.chunk_idx]
                self.chunk_idx += 1  # 游标向后移一位
                
                # 4. 解析与执行动作
                current_qpos = self.robot.get_joint_positions().numpy()
                target_qpos_np = action[:7]
                dq = target_qpos_np - current_qpos
                
                # 仅在需要重规划时打印详细差异，避免刷屏
                if self.chunk_idx == 1:
                    print("current:", current_qpos)
                    print("target :", target_qpos_np)
                    print("delta  :", dq, "max_abs:", np.max(np.abs(dq)))

                target_qpos = torch.tensor(target_qpos_np, dtype=torch.float32)
                target_gripper = action[7]

                self.robot.update_desired_joint_positions(target_qpos)
                
                # 5. 夹爪控制逻辑
                                # 5. 夹爪控制逻辑
                should_open = bool(target_gripper > 0.5)
                if should_open != self._last_gripper_open:
                    if should_open:
                       
                        #self.gripper.grasp(speed=0.1, force=20, epsilon_inner=0.001, epsilon_outer=0.08)
                        self.gripper.grasp(speed=0.1, force=15.0)
                    else:
                        # 【核心修改】用 goto 闭合到接近 0 (比如 2mm) 取代 grasp
                        # 将容差放大到夹爪的最大物理行程（0.08米）
                        
                        self.gripper.goto(width=0.075, speed=0.1, force=20)
                    self._last_gripper_open = should_open 
                               
                """ should_open = bool(target_gripper > 0.5)
                if should_open != self._last_gripper_open:
                    if should_open:
                        # 张开：目标设为最大行程，非阻塞执行
                        self.gripper.goto(width=0.078, speed=0.1, force=20, blocking=False) 
                    else:
                        # 【核心改动】
                        # 1. 目标宽度设为 -0.01（负数）：它会由于物理阻力停在物体表面。
                        # 2. blocking=False：这是关键！它不会让程序停在这里等结果，
                        #    而是让底层驱动一直处于“正在去往目标”的活跃状态。
                        # 3. 这种状态下，libfranka 不会启动 grasp 丢失监测逻辑。
                        self.gripper.goto(width=-0.01, speed=0.1, force=40, blocking=False)
                        
                        # 4. 如果你的 Polymetis 版本不支持 width 为负，请设为 0.00001
                        # self.gripper.goto(width=0.00001, speed=0.1, force=40, blocking=False)
                        
                    self._last_gripper_open = should_open """
                # 6. 维持 10Hz 节拍
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