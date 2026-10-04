
from native_lib import preload_conda_cpp_runtime

preload_conda_cpp_runtime()

import time

import grpc
import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from omega_interface import Omega7Interface


def maybe_connect_gripper(ip_address="localhost", port=50052):
    try:
        gripper = GripperInterface(ip_address=ip_address, port=port)
        gripper.get_state()
        print("🦾 已连接夹爪服务。")
        return gripper
    except grpc.RpcError:
        print("⚠️ 未检测到夹爪服务，将跳过夹爪同步控制。")
        return None


def run_teleoperation():
    print("="*50)
    print("🚀 正在初始化实机遥操作引擎...")
    
    # 1. 初始化设备
    omega = Omega7Interface()
    robot = RobotInterface(ip_address="localhost")
    gripper = maybe_connect_gripper()
    
    # 2. 启动机器人阻抗控制 (这是最安全的模式)
    print("🤖 正在激活 FR3 笛卡尔阻抗模式...")
    robot.start_cartesian_impedance()
    time.sleep(1.0) # 等待控制器稳定

    # 3. 设定系统锚点 (Anchor)
    input("\n⚠️ [对齐] 请将 Omega 手柄放在中心舒适位置，然后按【回车键】记录锚点...")
    
    # 获取 Omega 初始状态
    o_pos_init, o_quat_init, o_gripper_init = omega.get_state()
    o_rot_init = R.from_quat(o_quat_init).as_matrix()
    
    # 获取 FR3 初始状态
    r_pos_init, r_quat_init = robot.get_ee_pose()
    r_rot_init = R.from_quat(r_quat_init.numpy()).as_matrix()

    # 4. 设定超参数
    K_SCALE = 3.0  # 空间放大 3 倍
    GRIPPER_MAX_WIDTH = 0.08 # FR3 夹爪最大宽度 (8cm)
    O_GRIPPER_RANGE = 5.0   # 假设 Omega 夹爪角度变化范围是 25 度，请根据你刚才测试的值微调
    smoothed_pos = r_pos_init.numpy().copy()
    smoothed_quat = R.from_quat(r_quat_init.numpy())
    alpha = 0.05
    beta = 0.05
    print("\n✨ 影子控制已启动！移动主手，FR3 将实时跟随。按 Ctrl+C 停止。")
    print("="*50)

    try:
        while True:
            # --- A. 读取主手实时数据 ---
            o_pos_curr, o_quat_curr, o_gripper_curr = omega.get_state()
            
            # --- B. 计算目标位置 (增量映射) ---
            target_pos = np.zeros(3)
            delta_pos = o_pos_curr - o_pos_init
            
            # 轴向交换区：如果你发现方向不对，在这里修改索引
            # 例如：如果手柄前推(X)，机器人左平移(Y)，则 target_pos[1] = ... + delta_pos[0]
            target_pos[0] = r_pos_init.numpy()[0] - K_SCALE * delta_pos[0]
            target_pos[1] = r_pos_init.numpy()[1] - K_SCALE * delta_pos[1]
            target_pos[2] = r_pos_init.numpy()[2] + K_SCALE * delta_pos[2]
            # --- C. 计算目标姿态 (相对旋转) ---
            o_rot_curr = R.from_quat(o_quat_curr).as_matrix()
            rel_rot = np.dot(np.linalg.inv(o_rot_init), o_rot_curr)
            target_rot = np.dot(r_rot_init, rel_rot)
            target_quat = R.from_matrix(target_rot).as_quat() # [x,y,z,w]
            smoothed_pos = alpha * target_pos + (1-alpha) * smoothed_pos
            o_rot_curr = R.from_quat(o_quat_curr).as_matrix()
            rel_rot = np.dot(np.linalg.inv(o_rot_init), o_rot_curr)
            target_rot_mat = np.dot(r_rot_init, rel_rot)
            current_target_R = R.from_matrix(target_rot_mat)
            times = [0, 1]
            rots = R.from_quat([smoothed_quat.as_quat(), current_target_R.as_quat()])
            slerp = Slerp(times, rots)
            smoothed_quat = slerp([beta])[0] # 取插值点
            robot.update_desired_ee_pose(
                position=torch.tensor(smoothed_pos, dtype=torch.float32),
                orientation=torch.tensor(smoothed_quat.as_quat(), dtype=torch.float32)
            )

            # --- E. 夹爪控制 ---
            # 简单的线性映射，注意调整你的具体角度值
            # 角度越小，闭合越紧
            gripper_target = (o_gripper_curr / O_GRIPPER_RANGE) * GRIPPER_MAX_WIDTH
            gripper_target = np.clip(gripper_target, 0.0, GRIPPER_MAX_WIDTH)
            if gripper is not None:
                gripper.goto(width=gripper_target, speed=0.1, force=0.1, blocking=False)

            # 维持约 100Hz 的控制频率
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n🛑 接收到停止指令，正在安全退出...")
    finally:
        omega.close()
        # 释放阻抗控制，让机器人原地锁定
        robot.terminate_current_policy()
        print("🔌 连接已断开，机器人已锁定。")

if __name__ == "__main__":
    run_teleoperation()
