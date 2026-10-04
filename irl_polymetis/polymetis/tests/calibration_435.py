
import os
import cv2
import json
import numpy as np

class CalibrateEyeToHand:
    def __init__(self, data_dir="calibration_data_435_left2"):
        self.data_dir = data_dir
        self.board_size = (11, 8)
        self.square_size = 0.010 
        self.camera_serial = '348122070707'
        import pyrealsense2 as rs
        print("🔄 读取 D435i 硬件内参...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.camera_serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intrinsics.fx, 0.0, intrinsics.ppx],
                           [0.0, intrinsics.fy, intrinsics.ppy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D = np.array(intrinsics.coeffs, dtype=np.float64)
        pipeline.stop() 

        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def extract_transform(self, matrix_4x4):
        R = matrix_4x4[:3, :3]
        t = matrix_4x4[:3, 3].reshape(3, 1)
        return R, t

    def run_calibration(self):
        R_gripper2base_list, t_gripper2base_list = [], []
        R_target2cam_list, t_target2cam_list = [], []
        
        sample_dirs = sorted([d for d in os.listdir(self.data_dir) if d.startswith("sample_")])
        valid_samples = 0
        
        for sample_name in sample_dirs:
            sample_path = os.path.join(self.data_dir, sample_name)
            img = cv2.imread(os.path.join(sample_path, "color.png"))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            with open(os.path.join(sample_path, "ee_pose.json"), 'r') as f:
                T_base_ee = np.array(json.load(f)["ee_pose_matrix"])
            
            ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                ret_pnp, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.K, self.D)
                
                if ret_pnp:
                    R_cam, _ = cv2.Rodrigues(rvec)
                    
                    # --- 核心区别：Eye-to-Hand 必须对机器人位姿求逆 ---
                    T_ee_base = np.linalg.inv(T_base_ee)
                    R_ee_base, t_ee_base = self.extract_transform(T_ee_base)
                    
                    R_gripper2base_list.append(R_ee_base)
                    t_gripper2base_list.append(t_ee_base)
                    R_target2cam_list.append(R_cam)
                    t_target2cam_list.append(tvec)
                    valid_samples += 1
                    
                    cv2.drawChessboardCorners(img, self.board_size, corners2, ret)
                    cv2.imwrite(os.path.join(sample_path, "detected_chess.png"), img)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R_cam2gripper
        T_base_cam[:3, 3] = t_cam2gripper.flatten()
        
        print("\n🎉 D435i (Eye-to-Hand) 标定完成！T_base_cam:")
        print(np.array_str(T_base_cam, precision=4, suppress_small=True))
        np.save("D435i_extrinsics_chess.npy", T_base_cam)
        with open("D435i_extrinsics_chess.json", 'w') as f:
            json.dump({"T_base_cam": T_base_cam.tolist()}, f, indent=4)
        print("💾 外参已保存为 D435_extrinsics_chess.json")
if __name__ == "__main__":
    calibrator = CalibrateEyeToHand()
    calibrator.run_calibration()