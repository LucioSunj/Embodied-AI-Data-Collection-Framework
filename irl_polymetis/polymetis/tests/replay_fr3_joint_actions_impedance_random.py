import argparse
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional


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
from native_lib import preload_conda_cpp_runtime

preload_conda_cpp_runtime()

import grpc
import numpy as np
import pandas as pd
import torch
from polymetis import GripperInterface, RobotInterface


DATA_DIR = Path(
    "/home/lsk/openpi/Experiments/carrot_pepper_corn_all/"
    "lerobot_datasets_carrot/data"
)
DEFAULT_PATTERN = "episode_pickup_carrot_*.parquet"
DEFAULT_CONTROL_HZ = 10.0
DEFAULT_RESET_TIME = 4.0
DEFAULT_PAUSE_BETWEEN_FILES = 2.0
DEFAULT_MAX_JOINT_STEP = 0.35
DEFAULT_OPEN_WIDTH = 0.075


def list_parquet_files(data_dir: Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    return [path for path in files if path.is_file()]


def random_file_order(
    files: list[Path],
    *,
    rng: random.Random,
    max_files: int,
) -> Iterable[Path]:
    played = 0
    while True:
        shuffled = list(files)
        rng.shuffle(shuffled)
        for path in shuffled:
            if max_files > 0 and played >= max_files:
                return
            played += 1
            yield path


def _as_float_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values: {array}")
    return array


def load_action_trajectory(
    parquet_file: Path,
    *,
    stride: int = 1,
    control_hz: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    df = pd.read_parquet(parquet_file)
    if "action" not in df.columns:
        raise KeyError(f"{parquet_file} does not contain an 'action' column")

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)

    action_rows = [_as_float_array(value, name="action") for value in df["action"]]
    if len(action_rows) < 2:
        raise ValueError(f"{parquet_file} has fewer than 2 action rows")
    if action_rows[0].shape[0] < 7:
        raise ValueError(f"{parquet_file} action rows must contain at least 7 joints")

    joint_actions = np.stack([row[:7] for row in action_rows]).astype(np.float32)
    gripper_actions = None
    if action_rows[0].shape[0] >= 8:
        gripper_actions = np.asarray([row[7] for row in action_rows], dtype=np.float32)

    if control_hz is not None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        segment_dts = np.full(len(joint_actions) - 1, 1.0 / control_hz, dtype=np.float32)
        return joint_actions, gripper_actions, segment_dts

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        timestamp_dts = np.diff(timestamps)
        valid_dts = timestamp_dts[np.isfinite(timestamp_dts) & (timestamp_dts > 0.0)]
        if len(valid_dts) > 0:
            median_dt = float(np.median(valid_dts))
            lower = max(1e-3, 0.25 * median_dt)
            upper = 4.0 * median_dt
            segment_dts = np.clip(timestamp_dts, lower, upper).astype(np.float32)
            return joint_actions, gripper_actions, segment_dts

    segment_dts = np.full(len(joint_actions) - 1, 1.0 / DEFAULT_CONTROL_HZ, dtype=np.float32)
    return joint_actions, gripper_actions, segment_dts


def validate_joint_steps(
    joint_actions: np.ndarray,
    max_joint_step: float,
    allow_large_jumps: bool,
):
    deltas = np.abs(np.diff(joint_actions, axis=0))
    max_delta = float(np.max(deltas)) if len(deltas) else 0.0
    if max_delta > max_joint_step:
        row, joint = np.unravel_index(np.argmax(deltas), deltas.shape)
        message = (
            f"Large action jump in row {row}->{row + 1}, joint {joint}: "
            f"{max_delta:.4f} rad > {max_joint_step:.4f} rad"
        )
        if allow_large_jumps:
            print(f"Warning: {message}")
        else:
            raise ValueError(message)


class FrankaJointActionImpedanceReplayer:
    def __init__(
        self,
        *,
        ip_address: str = "localhost",
        port: int = 50051,
        use_gripper: bool = True,
        gain_scale: float = 1.0,
        adaptive: bool = False,
        one_means_close: bool = False,
        open_width: float = DEFAULT_OPEN_WIDTH,
    ):
        print("Connecting to Franka robot...")
        self.robot = RobotInterface(ip_address=ip_address, port=port)
        self.gripper = GripperInterface(ip_address=ip_address) if use_gripper else None
        self.gain_scale = gain_scale
        self.adaptive = adaptive
        self.one_means_close = one_means_close
        self.open_width = open_width
        self._last_gripper_closed = None
        self._gripper_lock = threading.Lock()
        self._terminate_current_policy_quiet()
        print("Connected. Existing active policy was terminated if present.")
        if self.gripper is not None:
            semantics = "1=close, 0=open" if one_means_close else "1=open, 0=close"
            print(f"Gripper action semantics: {semantics}")

    def _terminate_current_policy_quiet(self):
        try:
            self.robot.terminate_current_policy(return_log=False)
        except Exception:
            pass

    def _start_joint_impedance_controller(self):
        current_qpos = self.robot.get_joint_positions()
        self.robot.start_joint_impedance(
            Kq=self.robot.Kq_default * self.gain_scale,
            Kqd=self.robot.Kqd_default * self.gain_scale,
            adaptive=self.adaptive,
        )
        self.robot.update_desired_joint_positions(current_qpos)
        print(
            "Joint impedance controller started "
            f"(gain_scale={self.gain_scale:.3f}, adaptive={self.adaptive})."
        )

    def _set_gripper_from_binary(self, gripper_value: float):
        if self.gripper is None:
            return
        closed = bool(gripper_value > 0.5)
        if not self.one_means_close:
            closed = not closed

        with self._gripper_lock:
            if closed == self._last_gripper_closed:
                return
            self._last_gripper_closed = closed

        try:
            if closed:
                self.gripper.grasp(speed=0.1, force=20.0)
            else:
                self.gripper.goto(width=self.open_width, speed=0.1, force=20.0)
        except Exception as exc:
            print(f"Warning: gripper command failed: {exc}")

    def _set_gripper_async(self, gripper_value: float):
        thread = threading.Thread(
            target=self._set_gripper_from_binary,
            args=(gripper_value,),
            daemon=True,
        )
        thread.start()

    def reset_to_initial_action(self, initial_action: np.ndarray, reset_time: float):
        self._terminate_current_policy_quiet()
        target = torch.tensor(initial_action[:7], dtype=torch.float32)
        print(f"Moving to initial action over {reset_time:.2f}s...")
        self.robot.move_to_joint_positions(target, time_to_go=reset_time)
        time.sleep(0.5)

    def replay_file(
        self,
        parquet_file: Path,
        *,
        stride: int = 1,
        control_hz: Optional[float] = None,
        reset_time: float = DEFAULT_RESET_TIME,
        pause_at_end: float = 0.5,
        max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
        allow_large_jumps: bool = False,
        replay_gripper: bool = True,
    ) -> dict:
        print(f"Loading trajectory: {parquet_file.name}")
        joint_actions, gripper_actions, segment_dts = load_action_trajectory(
            parquet_file,
            stride=stride,
            control_hz=control_hz,
        )
        validate_joint_steps(joint_actions, max_joint_step, allow_large_jumps)

        if gripper_actions is not None and replay_gripper:
            self._set_gripper_async(float(gripper_actions[0]))
        self.reset_to_initial_action(joint_actions[0], reset_time)

        self._start_joint_impedance_controller()
        print(
            f"Replaying {len(joint_actions)} action rows "
            f"({float(np.sum(segment_dts)):.2f}s source duration)..."
        )

        start_time = time.time()
        next_tick = time.time()
        try:
            for idx, target_np in enumerate(joint_actions):
                target_qpos = torch.tensor(target_np, dtype=torch.float32)
                try:
                    self.robot.update_desired_joint_positions(target_qpos)
                except grpc.RpcError:
                    print("Controller RPC error. Restarting joint impedance controller.")
                    self._start_joint_impedance_controller()
                    self.robot.update_desired_joint_positions(target_qpos)

                if gripper_actions is not None and replay_gripper:
                    self._set_gripper_async(float(gripper_actions[idx]))

                if idx < len(segment_dts):
                    next_tick += float(segment_dts[idx])
                    sleep_time = next_tick - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("Replay interrupted by user.")
            raise

        if pause_at_end > 0:
            time.sleep(pause_at_end)

        actual_qpos = self.robot.get_joint_positions().detach().cpu().numpy()
        final_error = float(np.max(np.abs(actual_qpos - joint_actions[-1])))
        elapsed = time.time() - start_time
        print(f"Done. elapsed={elapsed:.2f}s, final max joint error={final_error:.5f} rad")
        return {
            "file": str(parquet_file),
            "rows": len(joint_actions),
            "elapsed": elapsed,
            "final_max_joint_error": final_error,
        }

    def close(self):
        self._terminate_current_policy_quiet()
        print("Robot policy terminated.")


def replay_random_loop(
    *,
    data_dir: Path,
    pattern: str,
    seed: Optional[int],
    max_files: int,
    pause_between_files: float,
    replayer_kwargs: dict,
    replay_kwargs: dict,
):
    files = list_parquet_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files matched {data_dir / pattern}")

    print(f"Found {len(files)} trajectories in {data_dir}")
    rng = random.Random(seed)
    replayer = FrankaJointActionImpedanceReplayer(**replayer_kwargs)
    try:
        for parquet_file in random_file_order(files, rng=rng, max_files=max_files):
            replayer.replay_file(parquet_file, **replay_kwargs)
            if pause_between_files > 0:
                time.sleep(pause_between_files)
    finally:
        replayer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly replay joint action trajectories from data_joint_actions "
            "using Polymetis joint impedance control."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--ip-address", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of random files to replay. 0 means loop forever.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Override source action rate. By default, timestamp deltas are used.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--reset-time", type=float, default=DEFAULT_RESET_TIME)
    parser.add_argument("--pause-at-end", type=float, default=0.5)
    parser.add_argument(
        "--pause-between-files",
        type=float,
        default=DEFAULT_PAUSE_BETWEEN_FILES,
    )
    parser.add_argument("--gain-scale", type=float, default=1.0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--max-joint-step", type=float, default=DEFAULT_MAX_JOINT_STEP)
    parser.add_argument("--allow-large-jumps", action="store_true")
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument(
        "--one-means-close",
        action="store_true",
        help="Use old convention where action[7]=1 means close. Default is 1=open, 0=close.",
    )
    parser.add_argument("--open-width", type=float, default=DEFAULT_OPEN_WIDTH)
    return parser.parse_args()


def main():
    args = parse_args()
    replay_random_loop(
        data_dir=args.data_dir.expanduser(),
        pattern=args.pattern,
        seed=args.seed,
        max_files=args.max_files,
        pause_between_files=args.pause_between_files,
        replayer_kwargs={
            "ip_address": args.ip_address,
            "port": args.port,
            "use_gripper": not args.disable_gripper,
            "gain_scale": args.gain_scale,
            "adaptive": args.adaptive,
            "one_means_close": args.one_means_close,
            "open_width": args.open_width,
        },
        replay_kwargs={
            "stride": args.stride,
            "control_hz": args.control_hz,
            "reset_time": args.reset_time,
            "pause_at_end": args.pause_at_end,
            "max_joint_step": args.max_joint_step,
            "allow_large_jumps": args.allow_large_jumps,
            "replay_gripper": not args.disable_gripper,
        },
    )


if __name__ == "__main__":
    main()
