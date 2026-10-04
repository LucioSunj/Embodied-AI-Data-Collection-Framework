import cv2
import time
import torch
import numpy as np
import pyrealsense2 as rs
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R

class VisualTouchValidator:
    def __init__(self, extrinsics_path="/home/lsk/irl_polymetis/D405_extrinsics_chess1.npy"):
        # 1. 加载标定好的外参矩阵 T_ee_cam
        print("📂 正在加载外参矩阵...")
        try:
            self.T_ee_cam = np.load(extrinsics_path)
            print("✅ 外参加载成功！")
        except FileNotFoundError:
            print(f"❌ 找不到文件 {extrinsics_path}，请先完成标定！")
            exit()

        # 2. 初始化 FR3 机器人
        print("🤖 正在连接机器人...")
        self.robot = RobotInterface(ip_address="localhost")
        # 确保机器人处于初始阻抗模式
        self.robot.start_joint_impedance()

        # 3. 初始化 D405 相机并开启深度对齐 (极其关键！)
        print("📷 正在启动 D405 相机 (RGB-D 对齐模式)...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        # 启用彩色流和深度流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = self.pipeline.start(config)
        
        # 提取对齐后的彩色相机内参
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # 创建对齐对象，将深度图对齐到彩色图的视角
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # 状态变量
        self.target_pixel = None
        cv2.namedWindow("Click to Touch")
        cv2.setMouseCallback("Click to Touch", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标双击事件"""
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.target_pixel = (x, y)
            print(f"\n🎯 接收到目标像素: ({x}, {y})")

    def get_ee_pose_matrix(self):
        """获取当前末端位姿 T_base_ee"""
        pos, quat = self.robot.get_ee_pose()
        rmat = R.from_quat(quat.numpy()).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rmat
        T[:3, 3] = pos.numpy()
        return T, quat

    def run(self):
        print("\n" + "="*50)
        print("🚀 视觉点触测试已就绪！")
        print("操作指南：")
        print("1. 在弹出的画面中，【双击】你想让机械臂触碰的物体")
        print("2. 机械臂会先移动到目标上方 5cm 处 (安全悬停)")
        print("3. 按键盘 'Q' 退出")
        print("="*50 + "\n")

        try:
            while True:
                # 获取对齐后的帧
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                img = np.asanyarray(color_frame.get_data()).copy()

                # 如果接收到了双击指令
                if self.target_pixel is not None:
                    u, v = self.target_pixel
                    
                    # 获取该像素点的真实深度 (米)
                    depth_val = depth_frame.get_distance(u, v)
                    
                    if depth_val <= 0.0 or depth_val > 1.0: # D405 有效工作距离
                        print("⚠️ 深度数据无效或超出范围 (0-1m)，请重新点击！")
                        self.target_pixel = None
                        continue
                        
                    print(f"📏 测得深度: {depth_val:.3f} 米")

                    # ==========================================
                    # 核心数学转换
                    # ==========================================
                    # 1. 像素系 -> 相机系 P_cam (使用 rs 内置的反投影函数)
                    point_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth_val)
                    P_cam = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])

                    # 2. 相机系 -> 末端系 P_ee
                    P_ee = self.T_ee_cam @ P_cam

                    # 3. 末端系 -> 基座系 P_base
                    T_base_ee, current_quat = self.get_ee_pose_matrix()
                    P_base = T_base_ee @ P_ee

                    target_x, target_y, target_z = P_base[:3]
                    print(f"📍 计算得出基座目标坐标: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}")

                    # ==========================================
                    # 执行移动 (加入安全机制)
                    # ==========================================
                    # 安全策略：不改变当前的旋转姿态，仅平移。并且目标高度 Z 增加 5cm 作为悬停
                    hover_z = target_z + 0.05 
                    
                    # 软限位保护：防止砸坏桌面 (假设桌面 Z 约等于 0.02)
                    if hover_z < 0.03:
                        print("🛑 触发安全保护：目标点过低，拒绝执行！")
                    else:
                        print(f"🛸 正在移动至悬停点 (Z={hover_z:.3f})...")
                        target_pos_tensor = torch.tensor([target_x, target_y, hover_z], dtype=torch.float32)
                        self.robot.move_to_ee_pose(
                            position=target_pos_tensor,
                            orientation=current_quat,
                            time_to_go=2.0 # 2秒内平滑到达
                        )
                        print("✅ 到达悬停点！")

                    # 清空点击状态
                    self.target_pixel = None

                # 画面显示与交互
                cv2.circle(img, (320, 240), 2, (0, 255, 0), -1) # 画个中心准星
                cv2.imshow("Click to Touch", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("🔌 系统已关闭。")

if __name__ == "__main__":
    validator = VisualTouchValidator()
    validator.run()