import os
import glob
import time
import torch
import numpy as np
import pandas as pd
from polymetis import RobotInterface, GripperInterface

# 1. 轨迹数据所在路径
#DATA_DIR = "/home/lsk/.cache/huggingface/lerobot/lsk/fr3_gello_lerobot_carrot/data/chunk-000/"
DATA_DIR = "/home/lsk/dataset/fr3_gello_lerobot_carrot1/data_restored/"
# 控制频率（需要与你采集数据时的频率保持一致，默认为之前的 10Hz）
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

    def reset_to_initial_pose(self, initial_state):
        """在开始回放前，先让机械臂平滑移动到轨迹的起始位置"""
        print("⏳ 正在移动到轨迹的起始位置...")
        target_qpos_np = np.array(initial_state[:7], dtype=np.float32)
        target_qpos = torch.tensor(target_qpos_np, dtype=torch.float32)
        
        # 让机器人以默认配置回到初始点，避免瞬间跳跃导致报错
        self.robot.move_to_joint_positions(target_qpos, time_to_go=3.0)
        time.sleep(1.0) # 稳定一下
        
        # 夹爪初始化
        initial_gripper = initial_state[7]
        should_open = bool(initial_gripper > 0.5)
        if should_open:
            self.gripper.grasp(speed=0.1, force=15.0)
        else:
            self.gripper.goto(width=0.075, speed=0.1, force=20)
        self._last_gripper_open = should_open
        print("✅ 已到达初始位置")

    def replay_file(self, parquet_file):
        """回放单个 parquet 文件"""
        print(f"📄 加载轨迹文件: {os.path.basename(parquet_file)}")
        df = pd.read_parquet(parquet_file)
        
        # 获取初始帧
        initial_state = df.iloc[0]['action']
        self.reset_to_initial_pose(initial_state)

        # 启动底层控制器
        self._start_joint_controller()
        
        print("🚀 开始复现轨迹...")
        try:
            for index, row in df.iterrows():
                start_t = time.time()
                
                # 提取状态: 前7维是关节角，最后1维是夹爪状态
                state = row['action']
                target_qpos_np = np.array(state[:7], dtype=np.float32)
                target_gripper = state[7]
                
                target_qpos = torch.tensor(target_qpos_np, dtype=torch.float32)

                # 1. 更新关节位置
                self.robot.update_desired_joint_positions(target_qpos)
                
                # 2. 夹爪控制逻辑 (复用之前的控制策略)
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
            print("🛑 回放被手动中断")

        print("🏁 单条轨迹复现结束。\n")

    def close(self):
        try:
            self.robot.terminate_current_policy()
            print("🔌 已断开机器人控制")
        except Exception:
            pass

if __name__ == "__main__":
    # 查找目录下所有的 parquet 文件
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    parquet_files.sort() # 按文件名排序保证顺序
    
    if not parquet_files:
        print(f"❌ 在 {DATA_DIR} 下没有找到任何 parquet 文件！")
        exit()

    print(f"🔍 找到 {len(parquet_files)} 条轨迹数据。")
    
    replayer = FrankaTrajectoryReplayer()
    
    try:
        # 可以选择只循环播放第一条，或者遍历播放所有
        # 这里默认遍历播放所有轨迹
        for file in parquet_files:
            replayer.replay_file(file)
            
            # 播放完一条后，等待两秒再播放下一条
            time.sleep(2)
            
    finally:
        replayer.close()