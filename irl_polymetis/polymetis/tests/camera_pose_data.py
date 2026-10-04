import os
import cv2
import json
import time
import torch
import numpy as np
import pyrealsense2 as rs
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R

# 保持你原来的环境预加载逻辑
try:
    from native_lib import preload_conda_cpp_runtime
    preload_conda_cpp_runtime()
except ImportError:
    pass

class IntegratedCalibrationCollector:
    def __init__(self, camera_serial, save_dir="calibration_data"):
        self.save_dir = save_dir
        self.camera_serial = camera_serial
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 1. 初始化机器人
        print("🤖 正在连接 FR3 机器人...")
        self.robot = RobotInterface(ip_address="localhost")
        
        # 2. 计算并启动示教参数 (完全复刻你的 teach_franka.py)
        print("✨ 正在激活跟随示教模式...")
        teach_kq = torch.clamp(self.robot.Kq_default * 0.01, min=1.0)
        teach_kqd = torch.clamp(self.robot.Kqd_default * 0.35, min=0.2)
        
        self.robot.start_joint_impedance(Kq=teach_kq, Kqd=teach_kqd, adaptive=False)
        
        # 3. 初始化相机 (RealSense)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.camera_serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        
        self.sample_idx = 0

    def get_ee_pose_matrix(self):
        """读取当前末端 4x4 变换矩阵"""
        pos, quat = self.robot.get_ee_pose()
        rmat = R.from_quat(quat.numpy()).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rmat
        T[:3, 3] = pos.numpy()
        return T

    def run(self):
        print("\n" + "="*50)
        print("📸 标定数据采集系统 (集成跟随示教)")
        print("操作提示：")
        print("1. 手动拖拽机械臂 (它会跟随并停留在你松手的地方)")
        print("2. 键盘 'C': 记录当前图像 + 位姿")
        print("3. 键盘 'Q': 锁定位置并保存退出")
        print("="*50 + "\n")

        try:
            while True:
                # A. 核心跟随逻辑：实时更新期望位置，防止“松手复位”
                current_joint_pos = self.robot.get_joint_positions()
                self.robot.update_desired_joint_positions(current_joint_pos)

                # B. 相机图像处理
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                
                # 深拷贝防止内存死锁
                img = np.asanyarray(color_frame.get_data()).copy()

                display_img = img.copy()
                cv2.putText(display_img, f"Samples: {self.sample_idx}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Calibration Tool", display_img)

                # C. 按键交互
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.save_sample(img)
                elif key == ord('q'):
                    break
                    
        finally:
            print("\n⏹️ 正在停止示教并锁定姿态...")
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            # 停止示教策略并恢复默认刚度锁定位置
            self.robot.terminate_current_policy()
            self.robot.start_joint_impedance(Kq=self.robot.Kq_default, Kqd=self.robot.Kqd_default)
            self.robot.update_desired_joint_positions(self.robot.get_joint_positions())
            print(f"✅ 采集完成，共 {self.sample_idx} 组数据。")

    def save_sample(self, img):
        sample_path = os.path.join(self.save_dir, f"sample_{self.sample_idx:03d}")
        os.makedirs(sample_path, exist_ok=True)
        cv2.imwrite(os.path.join(sample_path, "color.png"), img)
        pose_matrix = self.get_ee_pose_matrix()
        with open(os.path.join(sample_path, "ee_pose.json"), 'w') as f:
            json.dump({"ee_pose_matrix": pose_matrix.tolist()}, f, indent=4)
        print(f"📸 样本 {self.sample_idx} 已保存 (位置 + 图像)")
        self.sample_idx += 1

if __name__ == "__main__":
    # 使用你在南大实验室的 D435i 序列号
    collector = IntegratedCalibrationCollector(camera_serial='348122070707')
    collector.run()