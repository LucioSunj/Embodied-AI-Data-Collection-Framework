import pygame
import time
import sys

def test_xbox_triggers():
    # 1. 初始化系统
    pygame.init()
    pygame.joystick.init()

    # 2. 检查设备连接
    if pygame.joystick.get_count() == 0:
        print("❌ 未检测到手柄！请确保 Xbox 手柄已通过蓝牙或 USB 连接。")
        return

    # 3. 挂载第一个手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    num_axes = joystick.get_numaxes()
    
    print("="*60)
    print(f"✅ 成功连接设备: {joystick.get_name()}")
    print(f"📊 该手柄共有 {num_axes} 个连续输入轴 (Axes)")
    print("="*60)
    print("👉 请分别按下左右扳机键 (LT / RT) 和推动摇杆，观察数值变化")
    print("💡 提示：在 Ubuntu 下，LT 通常是 Axis 2 或 4，RT 通常是 Axis 5")
    print("按 Ctrl+C 退出测试\n")

    try:
        while True:
            # 必须调用 pump() 才能让 pygame 从操作系统获取最新输入
            pygame.event.pump() 

            # 获取所有轴的实时数据
            axes_data = [joystick.get_axis(i) for i in range(num_axes)]
            
            # 将所有轴的数据格式化为一行，方便对比观察
            output_str = " | ".join([f"Axis {i}: {val:5.2f}" for i, val in enumerate(axes_data)])
            
            # 使用 \r 原地刷新输出
            sys.stdout.write(f"\r{output_str}")
            sys.stdout.flush()
            
            # 50Hz 的刷新率，足够肉眼观察且不占用 CPU
            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\n\n🛑 监听结束，已安全退出。")
    finally:
        pygame.quit()

if __name__ == "__main__":
    test_xbox_triggers()