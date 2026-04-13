from __future__ import annotations

import argparse
import sys
from pathlib import Path


def bootstrap_workspace_paths() -> None:
    example_dir = Path(__file__).resolve().parent
    repo_root = example_dir.parent.parent
    sys.path.insert(0, str(repo_root / "lite_lerobot_infra" / "src"))
    sys.path.insert(0, str(repo_root / "lerobot" / "src"))
    sys.path.insert(0, str(repo_root / "irl_polymetis" / "polymetis" / "python"))


bootstrap_workspace_paths()

from lite_lerobot_infra.cameras import RealSenseCamera, RealSenseCameraConfig
from lite_lerobot_infra.recorder import DatasetRecorder, DatasetRecorderConfig
from lite_lerobot_infra.robots import FrankaPolymetisConfig, FrankaPolymetisRobot
from lite_lerobot_infra.teleoperators import XboxTeleopConfig, XboxTeleoperator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record FR3 teleoperation data into a LeRobotDataset.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. local/franka_xbox_demo")
    parser.add_argument("--root", required=True, help="Local dataset directory")
    parser.add_argument("--task", required=True, help="Task instruction stored with each frame")
    parser.add_argument("--ip-address", default="localhost")
    parser.add_argument("--robot-port", type=int, default=50051)
    parser.add_argument("--gripper-port", type=int, default=50052)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--camera-serial", action="append", default=[], help="Repeat for each RealSense serial")
    parser.add_argument("--hardware-reset-cameras", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cameras = {}
    for idx, serial in enumerate(args.camera_serial):
        camera_name = f"cam_{idx}"
        cameras[camera_name] = RealSenseCamera(
            RealSenseCameraConfig(
                name=camera_name,
                serial=serial,
                fps=args.fps,
                hardware_reset=args.hardware_reset_cameras,
            )
        )

    robot = FrankaPolymetisRobot(
        FrankaPolymetisConfig(
            ip_address=args.ip_address,
            robot_port=args.robot_port,
            gripper_port=args.gripper_port,
            cameras=cameras,
        )
    )
    teleop = XboxTeleoperator(XboxTeleopConfig())
    recorder = DatasetRecorder(
        robot=robot,
        teleop=teleop,
        config=DatasetRecorderConfig(
            repo_id=args.repo_id,
            root=args.root,
            task=args.task,
            fps=args.fps,
            resume=args.resume,
            max_episodes=args.max_episodes,
        ),
    )
    recorder.run()


if __name__ == "__main__":
    main()
