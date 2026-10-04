import os
import glob
import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from polymetis import RobotInterface
from tqdm import tqdm

# ================= 配置区 =================
# 原始数据目录
DATA_DIR = "/home/lsk/irl_polymetis/lerobot_datasets_all_things/data/"
# 修复后的数据保存目录（防覆盖保护）
RESTORED_DIR = "/home/lsk/irl_polymetis/lerobot_datasets_all_things/data_restored/"

# Polymetis IP 配置 (用于加载逆运动学模型)
POLIMETIS_IP = "localhost"
POLIMETIS_PORT = 50051
# ==========================================

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 找不到原始数据目录: {DATA_DIR}")
        return
        
    os.makedirs(RESTORED_DIR, exist_ok=True)
    
    print("🔌 正在连接 Polymetis 以加载逆运动学(IK)模型...")
    try:
        robot = RobotInterface(ip_address=POLIMETIS_IP, port=POLIMETIS_PORT)
    except Exception as e:
        print("❌ 连接失败！请确保 Polymetis 服务正在运行 (可使用 launch_robot.py sim=true)")
        return
    print("✅ 运动学模型加载成功")

    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    print(f"📄 找到 {len(parquet_files)} 个 Parquet 文件，开始修复...\n")

    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        restored_path = os.path.join(RESTORED_DIR, filename)
        
        # 读取原始 Parquet
        df = pd.read_parquet(file_path)
        
        restored_actions = []
        
        # 使用 tqdm 显示单个文件的处理进度
        print(f"🔄 正在处理: {filename}")
        for index, row in tqdm(df.iterrows(), total=len(df), leave=False):
            # 获取当前实际的关节状态 (作为 IK 求解的种子，防止发生手肘翻转奇异点)
            # 注意：qpos 是 8 维 [q1..q7, gripper_state]
            current_qpos = np.array(row['qpos'], dtype=np.float32)
            rest_pose = torch.tensor(current_qpos[:7], dtype=torch.float32)
            
            # 获取原始保存的 Cartesian Action (7 维: x, y, z, roll, pitch, yaw, gripper_cmd)
            cart_action = np.array(row['action'], dtype=np.float32)
            
            # 1. 拆解笛卡尔坐标
            target_pos_np = cart_action[:3]
            target_rpy_np = cart_action[3:6]
            target_gripper = cart_action[6]
            
            # 2. 将 RPY (Roll, Pitch, Yaw) 转回 Quaternion (x, y, z, w)
            # 使用 scipy，对应原始脚本中的 XYZ 内旋顺序
            quat_np = R.from_euler('xyz', target_rpy_np).as_quat()
            
            target_pos = torch.tensor(target_pos_np, dtype=torch.float32)
            target_quat = torch.tensor(quat_np, dtype=torch.float32)
            
            # 3. 调用逆运动学 (IK) 求解
            # 传入 rest_pose 确保求解出的关节角最接近机械臂当前姿态
            ik_solution = robot.robot_model.inverse_kinematics(
                target_pos, 
                target_quat, 
                rest_pose=rest_pose
            )
            
            # 4. 拼接回 8-DoF (7关节 + 1夹爪)
            ik_joints_np = ik_solution.detach().cpu().numpy()
            restored_action = np.append(ik_joints_np, target_gripper).astype(np.float32)
            
            restored_actions.append(restored_action.tolist())
            
        # 替换 DataFrame 中的 action 列
        df['action'] = restored_actions
        
        # 保存到新目录
        df.to_parquet(restored_path, engine="pyarrow")
        
    print(f"\n🎉 修复完成！所有转换后的数据已保存在: {RESTORED_DIR}")
    #print("你可以直接使用这个新目录进行 pi0 模型的训练了。")

if __name__ == "__main__":
    main()