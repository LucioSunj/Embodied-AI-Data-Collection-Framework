import time
import numpy as np
import pyrealsense2 as rs
import cv2
from polymetis import RobotInterface, GripperInterface
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import torch

# ================= 配置区 =================
# 1. 相机序列号映射 (已修改为 DROID 原生模型格式)
CAM_MAPPING = {
    '348122070707': 'observation/exterior_image_1_left',  # 主视角/侧视角相机
    '352122270841': 'observation/wrist_image_left',       # 腕部相机
    # '347622075736': 'observation/exterior_image_2_left' # 若不需要第三路可注释掉
}

# 2. 任务指令 (DROID 模型请务必使用准确的英文描述)
TASK_PROMPT = "pick up the corn"

# 3. 控制与规划参数
CONTROL_HZ = 10      # 整体控制循环与相机的更新频率
REPLAN_INTERVAL = 4  # 每次拿到新的 Chunk 后，连续盲跑（开环执行）的步数
MAX_DELTA = 0.05     # [安全防火墙] 单步允许的最大关节角变化量 (弧度)

# 4. 摄像头预览开关 (强制设为 False 以修复无显示器环境下的 Qt Core Dump 报错)
SHOW_CAMERA_PREVIEW = False
PREVIEW_CAMERA_SERIAL = "352122270841"
PREVIEW_WINDOW_NAME = f"RealSense Preview - {PREVIEW_CAMERA_SERIAL}"
# ==========================================

class FrankaPi0Deployer:
    def __init__(self):
        # 初始化模型客户端 (注意：如果你的 Server 在别处，请修改 IP)
        self.client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)
        
        # 初始化 Franka 与夹爪
        self.robot = RobotInterface(ip_address="localhost")
        self.gripper = GripperInterface(ip_address="localhost")
        
        # 初始化相机管线
        self.pipelines = {}
        self._last_gripper_open = None
        
        # 初始化 chunk 状态管理器
        self.action_chunk = None
        self.chunk_idx = 0
        
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
        # DROID 要求的文本指令键名是 "prompt"
        obs = {"prompt": TASK_PROMPT}
        
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

            # RGB 转换和 224 缩放，匹配 DROID 训练分布
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs[key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_rgb, 224, 224)
            )

        # 2. 获取机器人状态并装入 DROID 要求的键值
        qpos = self.robot.get_joint_positions().numpy()
        gripper_state = self.gripper.get_state()
        gripper_binary = 1.0 if gripper_state.width > 0.05 else 0.0
        
        # DROID 专用状态名拆分
        obs["observation/joint_position"] = np.asarray(qpos, dtype=np.float32)
        obs["observation/gripper_position"] = np.asarray([gripper_binary], dtype=np.float32)
        
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
        """非常重要：必须先启动阻抗控制器，才能接收 update 指令"""
        current_qpos = self.robot.get_joint_positions()
        self.robot.start_joint_impedance(
            Kq=self.robot.Kq_default,
            Kqd=self.robot.Kqd_default,
            adaptive=False,
        )
        self.robot.update_desired_joint_positions(current_qpos)
        print("✅ 关节阻抗控制器已成功启动")

    def _should_replan(self):
        """判断是否需要向大模型服务器请求新的 Action Chunk"""
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
            # 必须在主循环前启动控制器
            self._start_joint_controller()
            
            while True:
                start_t = time.time()
                
                # 1. 采集当前观察 (每次循环都必须调用，以冲刷相机缓存保持实时)
                obs = self.get_observation()
                
                # 2. 判断是否触发推理
                if self._should_replan():
                    result = self.client.infer(obs)
                    if "actions" not in result:
                        raise KeyError(f"服务端响应缺少 'actions': {result.keys()}")
                    
                    self.action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    self.chunk_idx = 0
                    print(f"\n🔄 Replan Triggered! 拿到新预测轨迹，开始执行...")

                # 3. 从现有的 Chunk 中提取当前步的动作
                action = self.action_chunk[self.chunk_idx]
                self.chunk_idx += 1  # 游标向后移一位
                
                # =======================================================
                # 4. 速度积分与安全防火墙 (DROID Velocity -> Position 逻辑)
                # =======================================================
                current_qpos = self.robot.get_joint_positions().numpy()
                
                # 提取网络输出的关节速度 (Joint Velocity)
                joint_velocity = action[:7]
                
                # 计算时间步长 dt
                dt = 1.0 / CONTROL_HZ
                
                # 速度转增量
                dq = joint_velocity * dt
                
                # 执行绝对安全截断 (防止突变导致 Reflex 锁死)
                dq_clipped = np.clip(dq, -MAX_DELTA, MAX_DELTA)
                
                # 增量叠加至当前真实位置
                target_qpos_safe = current_qpos + dq_clipped

                # 仅在重规划的第 1 步打印详细差异，避免刷屏
                if self.chunk_idx == 1:
                    print(f"current:        {np.round(current_qpos, 3)}")
                    print(f"model_velocity: {np.round(joint_velocity, 3)}")
                    print(f"delta_clipped:  {np.round(dq_clipped, 3)}")
                    print(f"target_safe:    {np.round(target_qpos_safe, 3)}")

                # 转换为 Tensor 并下发指令
                target_qpos = torch.tensor(target_qpos_safe, dtype=torch.float32)
                self.robot.update_desired_joint_positions(target_qpos)
                
                # 5. 夹爪控制逻辑 (绝对位置控制: 0 为闭合, 1 为张开)
                target_gripper = action[7]
                should_open = bool(target_gripper > 0.5)
                if should_open != self._last_gripper_open:
                    if should_open:
                        self.gripper.goto(width=0.08, speed=0.1, force=20) 
                    else:
                        self.gripper.grasp(speed=0.1, force=20) 
                    self._last_gripper_open = should_open

                # 6. 维持稳定的循环节拍
                sleep_time = (1.0 / CONTROL_HZ) - (time.time() - start_t)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 接收到中断信号，停止运行并释放机器人。")
        finally:
            self.close()

if __name__ == "__main__":
    deployer = FrankaPi0Deployer()
    deployer.run()