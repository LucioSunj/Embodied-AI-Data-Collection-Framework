import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from native_lib import preload_conda_cpp_runtime


def _ensure_conda_prefix_matches_python():
    python_path = Path(sys.executable).resolve()
    if python_path.parent.name != "bin":
        return

    inferred_prefix = python_path.parent.parent
    if (inferred_prefix / "conda-meta").exists():
        os.environ["CONDA_PREFIX"] = str(inferred_prefix)
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        env_lib = str(inferred_prefix / "lib")
        if env_lib not in current_ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (
                env_lib if not current_ld_path else f"{env_lib}:{current_ld_path}"
            )


_ensure_conda_prefix_matches_python()
preload_conda_cpp_runtime()

import numpy as np
import pandas as pd
import torch
from polymetis import RobotInterface
from tqdm import tqdm


DATA_DIR = Path(
    "/home/lsk/openpi/Experiments/carrot_pepper_corn_all/"
    "lerobot_datasets_green_pepper/data"
)
OUTPUT_DIR = DATA_DIR.parent / "data_joint_actions"

DEFAULT_START_EPISODE = 0
DEFAULT_END_EPISODE = 50
DEFAULT_PATTERN = "episode_pickup_green_pepper_{episode:06d}.parquet"

POLYMETIS_IP = "localhost"
POLYMETIS_PORT = 50051
GRIPPER_OPEN_THRESHOLD = 0.05
IK_TOLERANCE = 1e-3
IK_MAX_ITERS = 2000
IK_DT = 0.1
IK_DAMPING = 1e-6
SUCCESS_TOLERANCE = None
FALLBACK_POSE_ERROR_THRESHOLD = None
FALLBACK_MODE = "current_qpos"
MAX_QPOS_JOINT_DIFF = 0.35
MAX_PREVIOUS_ACTION_JOINT_DIFF = 0.25


def _normalize_quat(quat):
    quat = np.asarray(quat, dtype=np.float32)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-6:
        raise ValueError(f"Invalid near-zero quaternion: {quat.tolist()}")
    return quat / quat_norm


def _quat_angle_error(quat_a, quat_b):
    quat_a = _normalize_quat(quat_a)
    quat_b = _normalize_quat(quat_b)
    # q and -q represent the same orientation.
    dot = float(abs(np.dot(quat_a, quat_b)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _ik_pose_error(robot: RobotInterface, joint_pos, target_pos_np, target_quat_np):
    joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
    pos_output, quat_output = robot.robot_model.forward_kinematics(joint_pos)
    pos_output = pos_output.detach().cpu().numpy()
    quat_output = quat_output.detach().cpu().numpy()

    pos_err = float(np.linalg.norm(pos_output - target_pos_np))
    rot_err = float(_quat_angle_error(quat_output, target_quat_np))
    return float(np.linalg.norm([pos_err, rot_err])), pos_err, rot_err


def _solve_ik(
    robot: RobotInterface,
    target_pos_np,
    target_quat_np,
    seed_qpos,
    *,
    ik_tolerance: float,
    ik_max_iters: int,
    ik_dt: float,
    ik_damping: float,
):
    target_pos = torch.tensor(target_pos_np, dtype=torch.float32)
    target_quat = torch.tensor(target_quat_np, dtype=torch.float32)
    seed_pose = torch.tensor(seed_qpos[:7], dtype=torch.float32)

    ik_solution = robot.robot_model.inverse_kinematics(
        target_pos,
        target_quat,
        rest_pose=seed_pose,
        eps=ik_tolerance,
        max_iters=ik_max_iters,
        dt=ik_dt,
        damping=ik_damping,
    )
    joint_action = ik_solution.detach().cpu().numpy().astype(np.float32)
    pose_err, pos_err, rot_err = _ik_pose_error(
        robot,
        joint_action,
        target_pos_np,
        target_quat_np,
    )
    return joint_action, pose_err, pos_err, rot_err


def _append_unique_seed(seed_candidates, seed_qpos):
    seed_qpos = np.asarray(seed_qpos[:7], dtype=np.float32)
    for existing_seed in seed_candidates:
        if np.allclose(existing_seed, seed_qpos, atol=1e-5):
            return
    seed_candidates.append(seed_qpos)


def quat_action_to_joint_action(
    robot: RobotInterface,
    cart_action,
    rest_qpos,
    *,
    gripper_open_threshold: float = GRIPPER_OPEN_THRESHOLD,
    ik_tolerance: float = IK_TOLERANCE,
    ik_max_iters: int = IK_MAX_ITERS,
    ik_dt: float = IK_DT,
    ik_damping: float = IK_DAMPING,
    success_tolerance: Optional[float] = SUCCESS_TOLERANCE,
    previous_solution=None,
    fallback_pose_error_threshold: Optional[float] = FALLBACK_POSE_ERROR_THRESHOLD,
    fallback_mode: str = FALLBACK_MODE,
    max_qpos_joint_diff: Optional[float] = MAX_QPOS_JOINT_DIFF,
    max_previous_action_joint_diff: Optional[float] = MAX_PREVIOUS_ACTION_JOINT_DIFF,
) -> tuple[list[float], bool, float, float, float, bool, Optional[str]]:
    """Convert [x, y, z, qx, qy, qz, qw, gripper_width] to [q1..q7, gripper_binary]."""
    cart_action = np.asarray(cart_action, dtype=np.float32)
    rest_qpos = np.asarray(rest_qpos, dtype=np.float32)

    if cart_action.shape[0] < 8:
        raise ValueError(
            "action must be [x, y, z, qx, qy, qz, qw, gripper_width], "
            f"got shape {cart_action.shape}"
        )
    if rest_qpos.shape[0] < 7:
        raise ValueError(f"rest_qpos must contain at least 7 joints, got {rest_qpos.shape}")

    target_pos_np = cart_action[:3]
    target_quat_np = _normalize_quat(cart_action[3:7])

    gripper_width = float(cart_action[7])
    gripper_binary = 1.0 if gripper_width > gripper_open_threshold else 0.0

    seed_candidates = []
    rest_joint_qpos = rest_qpos[:7].astype(np.float32)
    previous_joint_action = None
    if previous_solution is not None:
        previous_solution = np.asarray(previous_solution, dtype=np.float32)
        if previous_solution.shape[0] >= 7:
            previous_joint_action = previous_solution[:7].astype(np.float32)

    _append_unique_seed(seed_candidates, rest_joint_qpos)
    if previous_joint_action is not None:
        for previous_weight in (0.25, 0.5, 0.75):
            blended_seed = (
                (1.0 - previous_weight) * rest_joint_qpos
                + previous_weight * previous_joint_action
            )
            _append_unique_seed(seed_candidates, blended_seed)
        _append_unique_seed(seed_candidates, previous_joint_action)

    success_threshold = ik_tolerance if success_tolerance is None else success_tolerance
    selection_pose_threshold = (
        success_threshold
        if fallback_pose_error_threshold is None
        else fallback_pose_error_threshold
    )

    best_joint_action = None
    best_pose_err = float("inf")
    best_pos_err = float("inf")
    best_rot_err = float("inf")
    best_score = None
    for seed_qpos in seed_candidates:
        joint_action, pose_err, pos_err, rot_err = _solve_ik(
            robot,
            target_pos_np,
            target_quat_np,
            seed_qpos,
            ik_tolerance=ik_tolerance,
            ik_max_iters=ik_max_iters,
            ik_dt=ik_dt,
            ik_damping=ik_damping,
        )
        qpos_joint_diff = float(np.max(np.abs(joint_action[:7] - rest_joint_qpos)))
        if previous_joint_action is None:
            prev_joint_diff = 0.0
        else:
            prev_joint_diff = float(
                np.max(np.abs(joint_action[:7] - previous_joint_action))
            )
        qpos_violation = (
            0.0
            if max_qpos_joint_diff is None
            else max(0.0, qpos_joint_diff - max_qpos_joint_diff)
        )
        prev_violation = (
            0.0
            if previous_joint_action is None
            or max_previous_action_joint_diff is None
            else max(0.0, prev_joint_diff - max_previous_action_joint_diff)
        )
        joint_cost = qpos_joint_diff + (
            0.0 if previous_joint_action is None else prev_joint_diff
        )
        score = (
            pose_err > selection_pose_threshold,
            qpos_violation + prev_violation > 0.0,
            qpos_violation + prev_violation,
            pose_err > success_threshold,
            joint_cost,
            pose_err,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_pose_err = pose_err
            best_pos_err = pos_err
            best_rot_err = rot_err
            best_joint_action = joint_action

    used_fallback = False
    fallback_reason = None
    if (
        fallback_pose_error_threshold is not None
        and best_pose_err > fallback_pose_error_threshold
    ):
        if fallback_mode == "current_qpos":
            best_joint_action = rest_qpos[:7].astype(np.float32)
            used_fallback = True
            fallback_reason = "pose_error"
        elif fallback_mode == "previous":
            if previous_solution is None:
                best_joint_action = rest_qpos[:7].astype(np.float32)
            else:
                best_joint_action = np.asarray(previous_solution[:7], dtype=np.float32)
            used_fallback = True
            fallback_reason = "pose_error"
        elif fallback_mode != "none":
            raise ValueError(
                "fallback_mode must be one of: none, current_qpos, previous; "
                f"got {fallback_mode}"
            )

    qpos_joint_diff = float(np.max(np.abs(best_joint_action[:7] - rest_qpos[:7])))
    prev_joint_diff = 0.0
    if previous_joint_action is not None:
        prev_joint_diff = float(
            np.max(np.abs(best_joint_action[:7] - previous_joint_action))
        )

    violates_qpos = (
        max_qpos_joint_diff is not None
        and qpos_joint_diff > max_qpos_joint_diff
    )
    violates_previous = (
        previous_joint_action is not None
        and max_previous_action_joint_diff is not None
        and prev_joint_diff > max_previous_action_joint_diff
    )
    if violates_qpos or violates_previous:
        if previous_joint_action is None:
            best_joint_action = rest_qpos[:7].astype(np.float32)
            reason = "qpos_joint_diff_first_frame"
        else:
            best_joint_action = previous_joint_action.astype(np.float32)
            if violates_qpos and violates_previous:
                reason = "qpos_and_previous_joint_diff"
            elif violates_qpos:
                reason = "qpos_joint_diff"
            else:
                reason = "previous_joint_diff"
        used_fallback = True
        fallback_reason = (
            reason if fallback_reason is None else f"{fallback_reason}+{reason}"
        )

    if used_fallback:
        best_pose_err, best_pos_err, best_rot_err = _ik_pose_error(
            robot,
            best_joint_action,
            target_pos_np,
            target_quat_np,
        )

    restored_action = np.append(best_joint_action, gripper_binary).astype(np.float32)
    return (
        restored_action.tolist(),
        bool(best_pose_err < success_threshold),
        best_pose_err,
        best_pos_err,
        best_rot_err,
        used_fallback,
        fallback_reason,
    )


def convert_parquet_file(
    robot: RobotInterface,
    input_path: Path,
    output_path: Path,
    *,
    gripper_open_threshold: float = GRIPPER_OPEN_THRESHOLD,
    ik_tolerance: float = IK_TOLERANCE,
    ik_max_iters: int = IK_MAX_ITERS,
    ik_dt: float = IK_DT,
    ik_damping: float = IK_DAMPING,
    success_tolerance: Optional[float] = SUCCESS_TOLERANCE,
    fallback_pose_error_threshold: Optional[float] = FALLBACK_POSE_ERROR_THRESHOLD,
    fallback_mode: str = FALLBACK_MODE,
    max_qpos_joint_diff: Optional[float] = MAX_QPOS_JOINT_DIFF,
    max_previous_action_joint_diff: Optional[float] = MAX_PREVIOUS_ACTION_JOINT_DIFF,
) -> dict:
    """Replace the parquet action column with joint actions and write output_path."""
    df = pd.read_parquet(input_path)

    if "action" not in df.columns:
        raise KeyError(f"{input_path} does not contain an 'action' column")

    state_col = "qpos" if "qpos" in df.columns else "observation.state"
    if state_col not in df.columns:
        raise KeyError(f"{input_path} does not contain 'qpos' or 'observation.state'")

    restored_actions = []
    failed_ik = 0
    pose_errors = []
    pos_errors = []
    rot_errors = []
    fallback_count = 0
    fallback_reasons = {}
    previous_solution = None

    for row_idx, row in tqdm(df.iterrows(), total=len(df), leave=False):
        (
            restored_action,
            success,
            pose_err,
            pos_err,
            rot_err,
            used_fallback,
            fallback_reason,
        ) = (
            quat_action_to_joint_action(
                robot,
                row["action"],
                row[state_col],
                gripper_open_threshold=gripper_open_threshold,
                ik_tolerance=ik_tolerance,
                ik_max_iters=ik_max_iters,
                ik_dt=ik_dt,
                ik_damping=ik_damping,
                success_tolerance=success_tolerance,
                previous_solution=previous_solution,
                fallback_pose_error_threshold=fallback_pose_error_threshold,
                fallback_mode=fallback_mode,
                max_qpos_joint_diff=max_qpos_joint_diff,
                max_previous_action_joint_diff=max_previous_action_joint_diff,
            )
        )
        restored_actions.append(restored_action)
        pose_errors.append(pose_err)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)
        if used_fallback:
            fallback_count += 1
            fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1
        if not success:
            failed_ik += 1
        previous_solution = np.asarray(restored_action[:7], dtype=np.float32)

    df["action"] = restored_actions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    worst_idx = int(np.argmax(pose_errors)) if pose_errors else -1
    return {
        "failed_ik": failed_ik,
        "max_pose_error": float(np.max(pose_errors)) if pose_errors else 0.0,
        "mean_pose_error": float(np.mean(pose_errors)) if pose_errors else 0.0,
        "max_pos_error": float(np.max(pos_errors)) if pos_errors else 0.0,
        "max_rot_error": float(np.max(rot_errors)) if rot_errors else 0.0,
        "worst_row": worst_idx,
        "fallback_count": fallback_count,
        "fallback_reasons": fallback_reasons,
    }


def iter_episode_files(
    data_dir: Path,
    *,
    start_episode: int,
    end_episode: int,
    pattern: str,
) -> list[Path]:
    files = []
    for episode in range(start_episode, end_episode + 1):
        path = data_dir / pattern.format(episode=episode)
        if path.exists():
            files.append(path)
        else:
            print(f"Missing file, skipped: {path}")
    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert LeRobot parquet action column from "
            "[x, y, z, qx, qy, qz, qw, gripper_width] to "
            "[q1..q7, gripper_binary]."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-episode", type=int, default=DEFAULT_START_EPISODE)
    parser.add_argument("--end-episode", type=int, default=DEFAULT_END_EPISODE)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--ip-address", default=POLYMETIS_IP)
    parser.add_argument("--port", type=int, default=POLYMETIS_PORT)
    parser.add_argument("--gripper-open-threshold", type=float, default=GRIPPER_OPEN_THRESHOLD)
    parser.add_argument("--ik-tolerance", type=float, default=IK_TOLERANCE)
    parser.add_argument("--ik-max-iters", type=int, default=IK_MAX_ITERS)
    parser.add_argument("--ik-dt", type=float, default=IK_DT)
    parser.add_argument("--ik-damping", type=float, default=IK_DAMPING)
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=SUCCESS_TOLERANCE,
        help=(
            "Tolerance used only for reporting IK failures. Defaults to --ik-tolerance."
        ),
    )
    parser.add_argument(
        "--fallback-pose-error-threshold",
        type=float,
        default=FALLBACK_POSE_ERROR_THRESHOLD,
        help=(
            "If set, replace IK results whose pose error is above this value. "
            "Useful for a few unreachable edge-frame targets."
        ),
    )
    parser.add_argument(
        "--fallback-mode",
        choices=("none", "current_qpos", "previous"),
        default=FALLBACK_MODE,
    )
    parser.add_argument(
        "--max-qpos-joint-diff",
        type=float,
        default=MAX_QPOS_JOINT_DIFF,
        help=(
            "Per-joint max allowed difference in radians between restored IK "
            "joints and the same row's qpos[:7]. Use a negative value to disable."
        ),
    )
    parser.add_argument(
        "--max-previous-action-joint-diff",
        type=float,
        default=MAX_PREVIOUS_ACTION_JOINT_DIFF,
        help=(
            "Per-joint max allowed difference in radians between current restored "
            "joints and the previous output action. Use a negative value to disable."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input parquet files instead of writing to --output-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = iter_episode_files(
        data_dir,
        start_episode=args.start_episode,
        end_episode=args.end_episode,
        pattern=args.pattern,
    )
    if not files:
        raise FileNotFoundError(f"No parquet files matched in {data_dir}")

    print("Connecting to Polymetis robot model for IK...")
    robot = RobotInterface(ip_address=args.ip_address, port=args.port)
    print("Connected.")

    total_failed_ik = 0
    total_fallback_count = 0
    total_fallback_reasons = {}
    max_qpos_joint_diff = (
        None if args.max_qpos_joint_diff < 0 else args.max_qpos_joint_diff
    )
    max_previous_action_joint_diff = (
        None
        if args.max_previous_action_joint_diff < 0
        else args.max_previous_action_joint_diff
    )
    for input_path in files:
        output_path = input_path if args.in_place else output_dir / input_path.name
        print(f"Converting {input_path.name} -> {output_path}")
        failed_ik = convert_parquet_file(
            robot,
            input_path,
            output_path,
            gripper_open_threshold=args.gripper_open_threshold,
            ik_tolerance=args.ik_tolerance,
            ik_max_iters=args.ik_max_iters,
            ik_dt=args.ik_dt,
            ik_damping=args.ik_damping,
            success_tolerance=args.success_tolerance,
            fallback_pose_error_threshold=args.fallback_pose_error_threshold,
            fallback_mode=args.fallback_mode,
            max_qpos_joint_diff=max_qpos_joint_diff,
            max_previous_action_joint_diff=max_previous_action_joint_diff,
        )
        total_failed_ik += failed_ik["failed_ik"]
        total_fallback_count += failed_ik["fallback_count"]
        for reason, count in failed_ik["fallback_reasons"].items():
            total_fallback_reasons[reason] = total_fallback_reasons.get(reason, 0) + count
        print(
            "  pose_error mean/max: "
            f"{failed_ik['mean_pose_error']:.6f} / {failed_ik['max_pose_error']:.6f}"
        )
        print(
            "  max pos/rot error: "
            f"{failed_ik['max_pos_error']:.6f} m / "
            f"{failed_ik['max_rot_error']:.6f} rad "
            f"(worst row {failed_ik['worst_row']})"
        )
        if failed_ik["failed_ik"]:
            print(f"  Warning: {failed_ik['failed_ik']} rows did not meet success tolerance.")
        if failed_ik["fallback_count"]:
            print(f"  Applied fallback to {failed_ik['fallback_count']} rows.")
            print(f"  Fallback reasons: {failed_ik['fallback_reasons']}")

    if args.in_place:
        print(f"Done. Overwrote {len(files)} parquet files in {data_dir}")
    else:
        print(f"Done. Wrote {len(files)} converted parquet files to {output_dir}")
    print(f"Total reported IK failures: {total_failed_ik}")
    if total_fallback_count:
        print(f"Total fallback rows: {total_fallback_count}")
        print(f"Total fallback reasons: {total_fallback_reasons}")


if __name__ == "__main__":
    # Avoid inheriting stale OpenMP settings from other training shells.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
