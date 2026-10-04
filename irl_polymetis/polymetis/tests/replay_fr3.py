import os
import glob
import time
import torch
import numpy as np
import pandas as pd
from polymetis import RobotInterface, GripperInterface

# 1. 轨迹数据所在路径
#DATA_DIR = "/home/lsk/.cache/huggingface/lerobot/lsk/fr3_gello_lerobot_carrot/data/chunk-000/"
DATA_DIR = "/home/lsk/dataset/fr3_gello_lerobot/data/"
# 控制频率（需要与你采集数据时的频率保持一致）
CONTROL_HZ = 30 

class FrankaTrajectoryReplayer:
    def __init__(self):
        print("🔌 正在连接 Franka 机器人...")
        self.robot = RobotInterface(ip_address="localhost")
        self.gripper = GripperInterface(ip_address="localhost")
        self._last_gripper_open = None
        print("✅ 机器人连接成功")

    def _start_joint_controller(self):
        """启动关节阻抗控制，并将目标初始对齐到当前位置"""
        current_qpos = self.robot.get_joint_positions()
        self.robot.start_joint_impedance(
            Kq=self.robot.Kq_default,
            Kqd=self.robot.Kqd_default,
            adaptive=False,
        )
        self.robot.update_desired_joint_positions(current_qpos)
        print("🛡️ 关节阻抗控制器已启动")

    
    def replay_file(self, parquet_file):
        """回放单个 parquet 文件"""
        print(f"📄 加载轨迹文件: {os.path.basename(parquet_file)}")
        df = pd.read_parquet(parquet_file)
        
        # 获取初始帧
        #initial_state = df.iloc[0]['observation.state']
        initial_state = df.iloc[0]['action']
        self.reset_to_initial_pose(initial_state)

        # 启动底层控制器
        self._start_joint_controller()
        
        print("🚀 开始复现轨迹...")
        try:
            for index, row in df.iterrows():
                start_t = time.time()
                
                # 提取状态: 前7维是关节角，最后1维是夹爪状态
                #state = row['observation.state']
                state = row['action']
                target_qpos_np = np.array(state[:7], dtype=np.float32)
                target_gripper = state[7]
                
                target_qpos = torch.tensor(target_qpos_np, dtype=torch.float32)

                # 1. 更新关节位置
                self.robot.update_desired_joint_positions(target_qpos)
                
                # 2. 夹爪控制逻辑
                should_open = bool(target_gripper > 0.5)
                if should_open != self._last_gripper_open:
                    if should_open:
                        self.gripper.grasp(speed=0.1, force=15.0)
                    else:
                        self.gripper.goto(width=0.075, speed=0.1, force=20)
                    self._last_gripper_open = should_open 

                # 3. 维持节拍
                sleep_time = (1.0 / CONTROL_HZ) - (time.time() - start_t)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n🛑 当前轨迹回放被手动中断 (Ctrl+C)")
            # 终止当前策略，以便可以继续下一次输入
            self.robot.terminate_current_policy()

        print("🏁 轨迹复现结束。\n")

    def close(self):
        try:
            self.robot.terminate_current_policy()
            print("🔌 已断开机器人控制")
        except Exception:
            pass

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ 找不到数据目录: {DATA_DIR}")
        exit()

    replayer = FrankaTrajectoryReplayer()
    
    try:
        while True:
            # ✨ 新增：等待用户输入轨迹编号
            user_input = input("👉 请输入要复现的轨迹编号 (例如输入 16 代表 episode_000016.parquet)，输入 'q' 退出: ").strip()
            
            if user_input.lower() == 'q':
                print("👋 退出程序")
                break
                
            if not user_input.isdigit():
                print("⚠️ 输入格式错误，请输入纯数字编号。\n")
                continue
                
            # 自动补全为 6 位数字格式，拼凑文件名
            episode_num = int(user_input)
            target_filename = f"episode_gello_{episode_num:06d}.parquet"
            target_file_path = os.path.join(DATA_DIR, target_filename)
            
            if not os.path.exists(target_file_path):
                print(f"❌ 找不到文件: {target_file_path}\n")
                continue
                
            # 执行回放
            replayer.replay_file(target_file_path)
            
    finally:
        replayer.close()