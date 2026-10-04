import os
import cv2
import json
import numpy as np
import pyrealsense2 as rs
import numpy as np

class CheckerboardCalibrator:
    def __init__(self, data_dir="calibration_data_405"):
        self.data_dir = data_dir
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        print("🔍 成功从 D405 硬件读取到最新内参：")
        print(f" - 分辨率: {intrinsics.width}x{intrinsics.height}")
        print(f" - 焦距 (fx, fy): ({intrinsics.fx:.2f}, {intrinsics.fy:.2f})")
        print(f" - 光心 (cx, cy): ({intrinsics.ppx:.2f}, {intrinsics.ppy:.2f})")
        print(f" - 畸变模型: {intrinsics.model}")
        print(f" - 畸变系数: {intrinsics.coeffs}")
        
        # 棋盘格的【内角点】数量 (列数, 行数)。注意：是数内部黑白交界的十字交叉点，不是数方块！
        # 例如：一张 10x7 个方块的棋盘格，内角点数量是 (9, 6)
        self.board_size = (11, 8) 
        # 单个黑白方块的物理边长（单位：米）
        self.square_size = 0.030 
        
        
        self.K = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                    [0.0, intrinsics.fy, intrinsics.ppy],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

        self.D = np.array(intrinsics.coeffs, dtype=np.float64)
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def extract_transform(self, matrix_4x4):
        """拆分 4x4 矩阵为旋转矩阵 R 和平移向量 t"""
        R = matrix_4x4[:3, :3]
        t = matrix_4x4[:3, 3].reshape(3, 1)
        return R, t

    def run_calibration(self):
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []
        
        sample_dirs = sorted([d for d in os.listdir(self.data_dir) if d.startswith("sample_")])
        print(f"🔍 找到 {len(sample_dirs)} 组采集数据，开始棋盘格角点提取...")
        
        valid_samples = 0
        for sample_name in sample_dirs:
            sample_path = os.path.join(self.data_dir, sample_name)
            img_path = os.path.join(sample_path, "color.png")
            pose_path = os.path.join(sample_path, "ee_pose.json")
            
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
                T_base_ee = np.array(pose_data["ee_pose_matrix"])
            
            # 1. 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
            
            if ret:
                # 2. 亚像素级精确化（大幅提升标定精度）
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                
                # 3. 使用 solvePnP 计算相机到标定板的外参 (T_cam_target)
                ret_pnp, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.K, self.D)
                
                if ret_pnp:
                    # 将旋转向量 (Rodrigues) 转换为 3x3 旋转矩阵
                    R_cam, _ = cv2.Rodrigues(rvec)
                    
                    R_ee, t_ee = self.extract_transform(T_base_ee)
                    
                    R_gripper2base_list.append(R_ee)
                    t_gripper2base_list.append(t_ee)
                    R_target2cam_list.append(R_cam)
                    t_target2cam_list.append(tvec)
                    valid_samples += 1
                    
                    # 在图像上画出角点并连线，保存用于 Debug
                    cv2.drawChessboardCorners(img, self.board_size, corners2, ret)
                    cv2.imwrite(os.path.join(sample_path, "detected_chess.png"), img)
                else:
                    print(f"⚠️ {sample_name}: solvePnP 解算位姿失败。")
            else:
                print(f"⚠️ {sample_name}: 未能检测到完整的 {self.board_size} 棋盘格，已跳过。")

        print(f"\n✅ 成功提取 {valid_samples} 组有效位姿对。")
            
        print("🧮 正在使用 Tsai-Lenz 算法进行 Eye-in-Hand 解算...")
        
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        T_ee_cam = np.eye(4)
        T_ee_cam[:3, :3] = R_cam2gripper
        T_ee_cam[:3, 3] = t_cam2gripper.flatten()
        
        print("\n🎉 标定完成！D405 相机相对于末端法兰的变换矩阵 T_ee_cam 为：")
        print(np.array_str(T_ee_cam, precision=4, suppress_small=True))
        
        np.save("D405_extrinsics_chess.npy", T_ee_cam)
        with open("D405_extrinsics_chess.json", 'w') as f:
            json.dump({"T_ee_cam": T_ee_cam.tolist()}, f, indent=4)
        print("💾 外参已保存为 D405_extrinsics_chess.json")

if __name__ == "__main__":
    
    calibrator = CheckerboardCalibrator()
    calibrator.run_calibration()