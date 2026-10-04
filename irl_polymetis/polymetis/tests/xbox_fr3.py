import pygame
import time
import torch
import numpy as np
import threading
import grpc  # 必须引入 grpc 以捕获服务器掉线异常
from polymetis import RobotInterface, GripperInterface

def apply_deadzone(val, deadzone=0.15):
    """带线性重映射的摇杆死区滤波器"""
    if abs(val) <= deadzone:
        return 0.0
    sign = np.sign(val)
    return sign * (abs(val) - deadzone) / (1.0 - deadzone)

def run_xbox_teleop():
    print("="*60)
    print("🎮 正在初始化 Xbox 混合遥操作引擎 (工业抗扭版)...")
    
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("❌ 未检测到手柄！请确保 Xbox 手柄已连接。")
        return
        
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    robot = RobotInterface(ip_address="localhost")
    gripper = GripperInterface(ip_address="localhost")
    
    # 封装一个重启控制器的辅助函数
    def start_controller():
        print("🤖 正在激活 Cartesian Impedance 控制器...")
        robot.start_cartesian_impedance(stiffness=[200, 200, 200, 20, 20, 20], damping_ratio=1.0)
        time.sleep(1.0)
        
    start_controller()
    
    current_pos, current_quat = robot.get_ee_pose()
    target_pos = current_pos.numpy().copy()
    
    MAX_VEL = 0.06           
    DT = 0.01                
    DEADZONE = 0.15          
    GRIPPER_MAX_WIDTH = 0.08 
    GRIPPER_THRESHOLD = 0.002 
    MAX_STRETCH = 0.03       # 虚拟弹簧最大拉伸极限 (3厘米)
    
    last_gripper_width = GRIPPER_MAX_WIDTH
    LIMITS = {'x': [0.25, 0.75], 'y': [-0.45, 0.45], 'z': [0.10, 0.65]}

    print("\n✨ 系统已就绪！")
    print("⚠️ 提示：若操作过猛导致掉线，系统将自动重连，不会崩溃！")
    print("="*60)

    try:
        while True:
            loop_start_time = time.time()
            pygame.event.pump()
            
            # 1. 实时读取真实的物理位置
            real_pos, real_quat = robot.get_ee_pose()
            real_pos_np = real_pos.numpy()
            
            # 2. 读取手柄输入
            raw_lx = joystick.get_axis(0)
            raw_ly = joystick.get_axis(1)
            raw_ry = joystick.get_axis(4) 
            raw_rt = joystick.get_axis(5)
            
            vx = -apply_deadzone(raw_ly, DEADZONE) 
            vy = -apply_deadzone(raw_lx, DEADZONE) 
            vz = -apply_deadzone(raw_ry, DEADZONE)
            
            # 3. 更新虚拟目标点
            target_pos[0] += vx * MAX_VEL * DT  
            target_pos[1] += vy * MAX_VEL * DT  
            target_pos[2] += vz * MAX_VEL * DT  
            
            target_pos[0] = np.clip(target_pos[0], LIMITS['x'][0], LIMITS['x'][1])
            target_pos[1] = np.clip(target_pos[1], LIMITS['y'][0], LIMITS['y'][1])
            target_pos[2] = np.clip(target_pos[2], LIMITS['z'][0], LIMITS['z'][1])

            # ==========================================
            # 核心修复 1：虚拟狗链 (Virtual Leash) 限幅
            # ==========================================
            dist = np.linalg.norm(target_pos - real_pos_np)
            if dist > MAX_STRETCH:
                # 强行将目标点往回拽，使其距离物理实体永远不超过 3cm
                target_pos = real_pos_np + (target_pos - real_pos_np) / dist * MAX_STRETCH

            # 4. 夹爪控制 (异步非阻塞)
            normalized_rt = (raw_rt + 1.0) / 2.0
            target_width = np.clip(GRIPPER_MAX_WIDTH * (1.0 - normalized_rt), 0.0, GRIPPER_MAX_WIDTH)

            if abs(target_width - last_gripper_width) > GRIPPER_THRESHOLD:
                def move_gripper_async(width_cmd):
                    try:
                        if width_cmd < 0.005:
                            gripper.grasp(speed=0.1, force=20.0)
                        else:
                            gripper.goto(width=width_cmd, speed=0.1, force=10.0)
                    except:
                        pass
                threading.Thread(target=move_gripper_async, args=(target_width,), daemon=True).start()
                last_gripper_width = target_width

            # ==========================================
            # 核心修复 2：防断联自动重连 (Auto-Recovery)
            # ==========================================
            try:
                robot.update_desired_ee_pose(
                    position=torch.tensor(target_pos, dtype=torch.float32),
                    orientation=current_quat # 保持抓取姿态向下
                )
            except grpc.RpcError as e:
                print("\n💥 检测到硬件保护 (Reflex) 触发，控制器已断开！")
                print("🔄 正在清除积分误差并自动重启...")
                
                # 同步坐标系：彻底消除误差积攒
                current_pos, current_quat = robot.get_ee_pose()
                target_pos = current_pos.numpy().copy()
                
                # 重启控制器
                start_controller()
                print("✅ 恢复成功，请轻推摇杆继续操作。")

            # 5. 退出与定频机制
            if joystick.get_button(6):
                break

            elapsed = time.time() - loop_start_time
            if DT > elapsed:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\n\n🛑 接收到系统中断信号")
    finally:
        print("🔌 正在锁定机器人并释放资源...")
        # 核心修复 3：防止在掉线状态下调用 terminate 导致二次崩溃
        try:
            robot.terminate_current_policy()
        except Exception:
            pass
        pygame.quit()
        print("✅ 安全退出完成。")

if __name__ == "__main__":
    run_xbox_teleop()