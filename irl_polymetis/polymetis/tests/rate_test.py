import time

from native_lib import preload_conda_cpp_runtime

preload_conda_cpp_runtime()

from polymetis import RobotInterface

# 1. 连接到你之前启动的 launch_robot.py (franka_sim)
try:
    robot = RobotInterface(ip_address="localhost")
    print("连接成功！开始测试...")
except Exception as e:
    print(f"连接失败，请确保 launch_robot.py 正在运行: {e}")
    exit()

# 2. 测量 1000 次获取状态的往返时间
latencies = []
for _ in range(1000):
    start = time.perf_counter()
    _ = robot.get_robot_state()  # 发起一次 gRPC 请求
    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # 转换为毫秒 (ms)

# 3. 计算结果
max_l = max(latencies)
avg_l = sum(latencies) / len(latencies)
over_1ms = len([l for l in latencies if l > 1.0])

print("-" * 30)
print(f"平均延迟 (Avg RTT): {avg_l:.3f} ms")
print(f"最大延迟 (Max RTT): {max_l:.3f} ms")
print(f"超过 1ms 的次数: {over_1ms} / 1000")
print("-" * 30)

if over_1ms > 100:
    print("结论：gRPC 层抖动严重。由于 RTT 经常超过 1ms，导致 Franka 丢包，这解释了 0.6 的成功率。")
