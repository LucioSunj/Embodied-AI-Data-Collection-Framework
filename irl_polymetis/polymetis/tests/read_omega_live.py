import time
import sys
from omega_interface import Omega7Interface

def monitor():
    try:
        omega = Omega7Interface()
        print("Omega 实时监控数据启动")
        print("="*50)

        while True:
            # 这里的解包顺序必须和 get_state 的 return 顺序完全一致
            pos, quat, gripper = omega.get_state()
            
            # 格式化输出
            # 如果 pos 是数组，pos[0] 就是 float，:6.3f 就能正常工作
            output = (
                f"\r📍 P: [{pos[0]:6.3f} {pos[1]:6.3f} {pos[2]:6.3f}] | "
                f"Q: [{quat[0]:5.2f} {quat[1]:5.2f} {quat[2]:5.2f} {quat[3]:5.2f}] | "
                f"G: {gripper:6.4f}"
            )
            
            sys.stdout.write(output)
            sys.stdout.flush()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n监测停止")
    finally:
        if 'omega' in locals(): omega.close()

if __name__ == "__main__":
    monitor()