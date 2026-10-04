import argparse
import time

from native_lib import preload_conda_cpp_runtime

preload_conda_cpp_runtime()

import torch

from polymetis import RobotInterface


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a timed hand-guiding session for a Franka arm in Polymetis."
    )
    parser.add_argument("--ip-address", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--duration",
        type=float,
        default=25.0,
        help="How long the hand-guiding window stays active, in seconds.",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=20.0,
        help="How often to refresh the desired joint target to the current pose.",
    )
    parser.add_argument(
        "--stiffness-scale",
        type=float,
        default=0.01,
        help="Scale applied to Polymetis default joint stiffness during teaching.",
    )
    parser.add_argument(
        "--damping-scale",
        type=float,
        default=0.35,
        help="Scale applied to Polymetis default joint damping during teaching.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=1.0,
        help="Seconds to wait after re-enabling normal stiffness at the final pose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    robot = RobotInterface(ip_address=args.ip_address, port=args.port)

    teach_kq = torch.clamp(robot.Kq_default * args.stiffness_scale, min=1.0)
    teach_kqd = torch.clamp(robot.Kqd_default * args.damping_scale, min=0.2)

    print(f"当前关节角度: {robot.get_joint_positions()}")
    print(
        f"进入示教模式 {args.duration:.1f} 秒。"
        "这段时间内你可以手动拖动机械臂，脚本会持续跟随并记录当前位置。"
    )
    print("示教结束后，机械臂会保持在你最后带到的位置。按 Ctrl+C 可提前结束。")

    robot.start_joint_impedance(Kq=teach_kq, Kqd=teach_kqd, adaptive=False)

    final_joint_pos = robot.get_joint_positions()
    start_time = time.time()
    update_period = 1.0 / max(args.update_hz, 1.0)

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                break

            final_joint_pos = robot.get_joint_positions()
            robot.update_desired_joint_positions(final_joint_pos)
            time.sleep(update_period)
    except KeyboardInterrupt:
        print("\n检测到提前结束，准备锁定当前姿态...")
        final_joint_pos = robot.get_joint_positions()
    finally:
        try:
            robot.terminate_current_policy(return_log=False)
        except Exception:
            pass

    print(f"示教结束，最终关节角度: {final_joint_pos}")
    print("正在锁定最终姿态...")
    robot.start_joint_impedance(
        Kq=robot.Kq_default,
        Kqd=robot.Kqd_default,
        adaptive=False,
    )
    robot.update_desired_joint_positions(final_joint_pos)
    time.sleep(max(args.hold_time, 0.5))
    print("已锁定到最终姿态。")


if __name__ == "__main__":
    main()
