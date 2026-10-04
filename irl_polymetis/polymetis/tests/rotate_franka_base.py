import argparse
import math
import time

from native_lib import preload_conda_cpp_runtime

preload_conda_cpp_runtime()

import torch

from polymetis import RobotInterface


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rotate Franka joint 1 (base joint) by a fixed angle, wait, and return."
    )
    parser.add_argument("--ip-address", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--degrees",
        type=float,
        default=-30.0,
        help="Rotation angle applied to the base joint, in degrees.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=5.0,
        help="How long to stay at the rotated pose before returning.",
    )
    parser.add_argument(
        "--time-to-go",
        type=float,
        default=None,
        help="Motion duration for each move, in seconds. Defaults to Polymetis adaptive timing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    robot = RobotInterface(ip_address=args.ip_address, port=args.port)

    start_joint_pos = robot.get_joint_positions().clone()
    target_joint_pos = start_joint_pos.clone()
    target_joint_pos[0] += math.radians(args.degrees)

    joint_limits_low, joint_limits_high = robot.robot_model.get_joint_angle_limits()
    if not (joint_limits_low[0] <= target_joint_pos[0] <= joint_limits_high[0]):
        raise ValueError(
            f"Base joint target {target_joint_pos[0].item():.4f} rad exceeds limits "
            f"[{joint_limits_low[0].item():.4f}, {joint_limits_high[0].item():.4f}] rad."
        )

    print(f"起始关节角度: {start_joint_pos}")
    print(
        f"正在将底座关节旋转 {args.degrees:.1f} 度 "
        f"({target_joint_pos[0].item():.4f} rad)..."
    )
    robot.move_to_joint_positions(target_joint_pos, time_to_go=args.time_to_go)

    print(f"已到目标姿态，保持 {args.wait_seconds:.1f} 秒...")
    time.sleep(args.wait_seconds)

    print("正在返回起始姿态...")
    robot.move_to_joint_positions(start_joint_pos, time_to_go=args.time_to_go)

    print(f"完成，当前关节角度: {robot.get_joint_positions()}")


if __name__ == "__main__":
    main()
