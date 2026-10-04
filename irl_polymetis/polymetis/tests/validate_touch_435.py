import cv2
import time
import torch
import numpy as np
import pyrealsense2 as rs
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R

class VisualTouchValidatorD435i:
    def __init__(self, extrinsics_path="D435_extrinsics_chess_left.npy", camera_serial="348122070707"):
        # 1. 加载眼在手外 (Eye-to-Hand) 的外参矩阵 T_base_cam
        print("📂 正在加载 D435i 外参矩阵...")
        try:
            # 这个矩阵直接将相机坐标系下的点，转换到机器人基座坐标系
            self.T_base_cam = np.load(extrinsics_path)
            print("✅ 外参加载成功！T_base_cam:")
            print(np.array_str(self.T_base_cam, precision=3, suppress_small=True))
        except FileNotFoundError:
            print(f"❌ 找不到文件 {extrinsics_path}，请先使用 D435i 数据完成 Eye-to-Hand 标定！")
            exit()

        # 2. 初始化 FR3 机器人
        print("🤖 正在连接机器人...")
        self.robot = RobotInterface(ip_address="localhost")
        self.robot.start_joint_impedance()

        # 3. 初始化 D435i 相机并开启深度对齐
        print(f"📷 正在启动 D435i 全局相机 [{camera_serial}]...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(camera_serial) # 指定 D435i 序列号
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = self.pipeline.start(config)
        
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.target_pixel = None
        cv2.namedWindow("D435i Global View - Double Click to Touch")
        cv2.setMouseCallback("D435i Global View - Double Click to Touch", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.target_pixel = (x, y)
            print(f"\n🎯 全局视野接收到目标像素: ({x}, {y})")

    def run(self):
        print("\n" + "="*50)
        print("🚀 D435i (Eye-to-Hand) 视觉验证已就绪！")
        print("操作指南：")
        print("1. 在画面中双击目标物体 (建议选一个有高度的物体，如水杯或积木)")
        print("2. 机械臂将移动至该点正上方 5cm 处")
        print("3. 按 'Q' 退出")
        print("="*50 + "\n")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                img = np.asanyarray(color_frame.get_data()).copy()

                if self.target_pixel is not None:
                    u, v = self.target_pixel
                    depth_val = depth_frame.get_distance(u, v)
                    
                    # D435i 是全局相机，距离通常比腕部相机远，放宽深度限制到 2.0 米
                    if depth_val <= 0.0 or depth_val > 2.0: 
                        print("⚠️ 深度数据无效或超出范围，请点击有效区域！")
                        self.target_pixel = None
                        continue
                        
                    print(f"📏 D435i 测得深度: {depth_val:.3f} 米")

                    
                    point_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth_val)
                    P_cam = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])

                    
                    P_base = self.T_base_cam @ P_cam
                    #P_base = np.linalg.inv(self.T_base_cam) @ P_cam
                    target_x, target_y, target_z = P_base[:3]
                    
                    print(f"📍 计算得出基座坐标: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}")

                    # 获取当前姿态，保持夹爪向下移动
                    _, current_quat = self.robot.get_ee_pose()

                    # 悬停与安全保护
                    hover_z = target_z + 0.05 
                    target_pos_tensor = torch.tensor([target_x, target_y, hover_z], dtype=torch.float32)
                    if hover_z < 0.03 or hover_z > 0.6:
                        print("🛑 安全保护：目标点过低！")
                        
                    else:
                        print(f"🛸 机械臂前往悬停点 (Z={hover_z:.3f})...")
                        self.robot.move_to_ee_pose(
                            position=target_pos_tensor ,
                            orientation=current_quat,
                            time_to_go=2.0 
                        )
                        print("✅ 到达目标上方！")

                    self.target_pixel = None

                cv2.imshow("D435i Global View - Double Click to Touch", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("🔌 系统已关闭。")

if __name__ == "__main__":
    # 请填入你 D435i 的真实序列号
    MY_D435I_SERIAL = "348122070707" #left
    #MY_D435I_SERIAL = "347622075736" #right
    validator = VisualTouchValidatorD435i(
        extrinsics_path="/home/lsk/irl_polymetis/D435i_extrinsics_chess_left.npy", 
        camera_serial=MY_D435I_SERIAL
    )
    validator.run()