from native_lib import preload_conda_cpp_runtime

preload_conda_cpp_runtime()

from polymetis import RobotInterface

# 1. 初始化机器人接口 (会自动寻找本地运行的服务器)
robot = RobotInterface(ip_address="localhost")

# 2. 获取当前关节角度
current_joint_pos = robot.get_joint_positions()
print(f"当前关节角度: {current_joint_pos}")

# 3. 简单的阻抗控制测试：让机器人保持在当前位置，但你可以用手轻轻推它
# (由于 Polymetis 默认就是力控/阻抗控制，你会发现它变得“软”了)
print("现在你可以尝试手动示教或移动机械臂了...")
