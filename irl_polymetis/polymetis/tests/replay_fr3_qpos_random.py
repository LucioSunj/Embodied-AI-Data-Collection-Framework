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

import numpy as np
import pandas as pd
import torch
import torchcontrol as toco
from polymetis import GripperInterface, RobotInterface


DATA_DIR = Path(
    "/home/lsk/openpi/Experiments/carrot_pepper_corn_all/"
    "lerobot_datasets_all_things/data"
)
DEFAULT_PATTERN = "episode_pickup_all_*.parquet"
DEFAULT_CONTROL_HZ = 10.0
DEFAULT_RESET_TIME = 4.0
DEFAULT_PAUSE_BETWEEN_FILES = 2.0
DEFAULT_MAX_JOINT_STEP = 0.35
DEFAULT_OPEN_WIDTH = 0.075
DEFAULT_ORDER = "random"


def list_parquet_files(data_dir: Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    return [path for path in files if path.is_file()]


def random_file_order(
    files: list[Path],
    *,
    rng: random.Random,
    repeat: bool,
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
        if not repeat:
            return


def sequential_file_order(
    files: list[Path],
    *,
    repeat: bool,
    max_files: int,
) -> Iterable[Path]:
    played = 0
    while True:
        for path in files:
            if max_files > 0 and played >= max_files:
                return
            played += 1
            yield path
        if not repeat:
            return


def _as_float_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values: {array}")
    return array


def load_qpos_trajectory(
    parquet_file: Path,
    *,
    stride: int = 1,
    control_hz: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    df = pd.read_parquet(parquet_file)
    if "qpos" not in df.columns:
        raise KeyError(f"{parquet_file} does not contain a 'qpos' column")

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)

    qpos_rows = [_as_float_array(value, name="qpos") for value in df["qpos"]]
    if len(qpos_rows) < 2:
        raise ValueError(f"{parquet_file} has fewer than 2 qpos rows")

    qpos = np.stack([row[:7] for row in qpos_rows]).astype(np.float32)
    gripper = None
    if qpos_rows[0].shape[0] >= 8:
        gripper = np.asarray([row[7] for row in qpos_rows], dtype=np.float32)

    if control_hz is not None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        segment_dts = np.full(len(qpos) - 1, 1.0 / control_hz, dtype=np.float32)
        return qpos, gripper, segment_dts

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        timestamp_dts = np.diff(timestamps)
        valid_dts = timestamp_dts[np.isfinite(timestamp_dts) & (timestamp_dts > 0.0)]
        if len(valid_dts) > 0:
            median_dt = float(np.median(valid_dts))
            lower = max(1e-3, 0.25 * median_dt)
            upper = 4.0 * median_dt
            segment_dts = np.clip(timestamp_dts, lower, upper).astype(np.float32)
            return qpos, gripper, segment_dts

    segment_dts = np.full(len(qpos) - 1, 1.0 / DEFAULT_CONTROL_HZ, dtype=np.float32)
    return qpos, gripper, segment_dts


def validate_joint_steps(qpos: np.ndarray, max_joint_step: float, allow_large_jumps: bool):
    deltas = np.abs(np.diff(qpos, axis=0))
    max_delta = float(np.max(deltas)) if len(deltas) else 0.0
    if max_delta > max_joint_step:
        row, joint = np.unravel_index(np.argmax(deltas), deltas.shape)
        message = (
            f"Large qpos jump in row {row}->{row + 1}, joint {joint}: "
            f"{max_delta:.4f} rad > {max_joint_step:.4f} rad"
        )
        if allow_large_jumps:
            print(f"Warning: {message}")
        else:
            raise ValueError(message)


def build_robot_hz_trajectory(
    qpos: np.ndarray,
    segment_dts: np.ndarray,
    robot_hz: float,
    *,
    final_hold_time: float = 0.2,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if len(qpos) != len(segment_dts) + 1:
        raise ValueError("qpos length must be one larger than segment_dts length")
    if robot_hz <= 0:
        raise ValueError(f"robot_hz must be positive, got {robot_hz}")

    pos_traj = []
    vel_traj = []
    for idx in range(len(qpos) - 1):
        dt = float(segment_dts[idx])
        steps = max(1, int(round(dt * robot_hz)))
        velocity = (qpos[idx + 1] - qpos[idx]) / max(dt, 1e-6)
        for step in range(steps):
            alpha = step / float(steps)
            desired_pos = (1.0 - alpha) * qpos[idx] + alpha * qpos[idx + 1]
            pos_traj.append(torch.tensor(desired_pos, dtype=torch.float32))
            vel_traj.append(torch.tensor(velocity, dtype=torch.float32))

    hold_steps = max(1, int(round(final_hold_time * robot_hz)))
    final_pos = torch.tensor(qpos[-1], dtype=torch.float32)
    zero_vel = torch.zeros(7, dtype=torch.float32)
    for _ in range(hold_steps):
        pos_traj.append(final_pos)
        vel_traj.append(zero_vel)

    return pos_traj, vel_traj


class FrankaQposRandomReplayer:
    def __init__(
        self,
        *,
        ip_address: str = "localhost",
        port: int = 50051,
        use_gripper: bool = True,
        gain_scale: float = 1.0,
        one_means_close: bool = False,
        open_width: float = DEFAULT_OPEN_WIDTH,
    ):
        print("Connecting to Franka robot...")
        self.robot = RobotInterface(ip_address=ip_address, port=port)
        self.gripper = GripperInterface(ip_address=ip_address) if use_gripper else None
        self.gain_scale = gain_scale
        self.one_means_close = one_means_close
        self.open_width = open_width
        self._last_gripper_closed = None
        self._terminate_current_policy_quiet()
        print("Connected. Existing active policy was terminated if present.")
        if self.gripper is not None:
            semantics = "1=close, 0=open" if one_means_close else "1=open, 0=close"
            print(f"Gripper qpos semantics: {semantics}")

    def _terminate_current_policy_quiet(self):
        try:
            self.robot.terminate_current_policy(return_log=False)
        except Exception:
            pass

    def _set_gripper_from_binary(self, gripper_value: float):
        if self.gripper is None:
            return
        closed = bool(gripper_value > 0.5)
        if not self.one_means_close:
            closed = not closed
        if closed == self._last_gripper_closed:
            return

        if closed:
            self.gripper.grasp(speed=0.1, force=20.0)
        else:
            self.gripper.goto(width=self.open_width, speed=0.1, force=20.0)
        self._last_gripper_closed = closed

    def reset_to_initial_qpos(self, initial_qpos: np.ndarray, reset_time: float):
        self._terminate_current_policy_quiet()
        target = torch.tensor(initial_qpos[:7], dtype=torch.float32)
        print(f"Moving to initial qpos over {reset_time:.2f}s...")
        self.robot.move_to_joint_positions(target, time_to_go=reset_time)
        time.sleep(0.5)

    def _send_policy_and_wait(self, torch_policy: toco.PolicyModule, timeout: float):
        self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)
        start_time = time.time()
        while self.robot.is_running_policy():
            if timeout is not None and time.time() - start_time > timeout:
                self._terminate_current_policy_quiet()
                raise TimeoutError(f"Trajectory replay exceeded timeout {timeout:.2f}s")
            time.sleep(0.02)

    def _start_gripper_replay_thread(self, gripper: np.ndarray, segment_dts: np.ndarray):
        if self.gripper is None:
            return None
        stop_event = threading.Event()

        def worker():
            self._set_gripper_from_binary(float(gripper[0]))
            for idx, value in enumerate(gripper[1:]):
                dt = float(segment_dts[min(idx, len(segment_dts) - 1)])
                if stop_event.wait(dt):
                    return
                self._set_gripper_from_binary(float(value))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread, stop_event

    def replay_file(
        self,
        parquet_file: Path,
        *,
        stride: int = 1,
        control_hz: Optional[float] = None,
        reset_time: float = DEFAULT_RESET_TIME,
        final_hold_time: float = 0.2,
        max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
        allow_large_jumps: bool = False,
        replay_gripper: bool = True,
    ) -> dict:
        print(f"Loading trajectory: {parquet_file.name}")
        qpos, gripper, segment_dts = load_qpos_trajectory(
            parquet_file,
            stride=stride,
            control_hz=control_hz,
        )
        validate_joint_steps(qpos, max_joint_step, allow_large_jumps)

        if gripper is not None and replay_gripper:
            self._set_gripper_from_binary(float(gripper[0]))
        self.reset_to_initial_qpos(qpos[0], reset_time)

        self._terminate_current_policy_quiet()
        robot_hz = float(getattr(self.robot, "hz", self.robot.metadata.hz))
        pos_traj, vel_traj = build_robot_hz_trajectory(
            qpos,
            segment_dts,
            robot_hz,
            final_hold_time=final_hold_time,
        )

        Kq = self.robot.Kq_default * self.gain_scale
        Kqd = self.robot.Kqd_default * self.gain_scale
        torch_policy = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=pos_traj,
            joint_vel_trajectory=vel_traj,
            Kq=Kq,
            Kqd=Kqd,
            Kx=self.robot.Kx_default,
            Kxd=self.robot.Kxd_default,
            robot_model=self.robot.robot_model,
            ignore_gravity=self.robot.use_grav_comp,
        )

        duration = float(np.sum(segment_dts)) + final_hold_time
        print(
            f"Replaying {len(qpos)} qpos rows as {len(pos_traj)} robot steps "
            f"({duration:.2f}s, robot_hz={robot_hz:.1f})..."
        )

        gripper_thread = None
        gripper_stop_event = None
        if gripper is not None and replay_gripper:
            gripper_thread, gripper_stop_event = self._start_gripper_replay_thread(
                gripper,
                segment_dts,
            )

        start_time = time.time()
        try:
            self._send_policy_and_wait(torch_policy, timeout=duration + 10.0)
        except Exception:
            if gripper_stop_event is not None:
                gripper_stop_event.set()
            raise
        finally:
            if gripper_thread is not None:
                gripper_thread.join(timeout=1.0)
                if gripper_thread.is_alive() and gripper_stop_event is not None:
                    gripper_stop_event.set()
        elapsed = time.time() - start_time

        actual_qpos = self.robot.get_joint_positions().detach().cpu().numpy()
        final_error = float(np.max(np.abs(actual_qpos - qpos[-1])))
        print(f"Done. elapsed={elapsed:.2f}s, final max joint error={final_error:.5f} rad")
        return {
            "file": str(parquet_file),
            "rows": len(qpos),
            "duration": duration,
            "elapsed": elapsed,
            "final_max_joint_error": final_error,
        }

    def close(self):
        self._terminate_current_policy_quiet()
        print("Robot policy terminated.")


def replay_random_qpos_loop(
    *,
    data_dir: Path = DATA_DIR,
    pattern: str = DEFAULT_PATTERN,
    order: str = DEFAULT_ORDER,
    repeat: bool = False,
    max_files: int = 0,
    seed: Optional[int] = None,
    **replayer_kwargs,
):
    files = list_parquet_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files matched {data_dir / pattern}")

    rng = random.Random(seed)
    replayer = FrankaQposRandomReplayer(
        ip_address=replayer_kwargs.pop("ip_address", "localhost"),
        port=replayer_kwargs.pop("port", 50051),
        use_gripper=not replayer_kwargs.pop("disable_gripper", False),
        gain_scale=replayer_kwargs.pop("gain_scale", 1.0),
        one_means_close=replayer_kwargs.pop("one_means_close", False),
        open_width=replayer_kwargs.pop("open_width", DEFAULT_OPEN_WIDTH),
    )

    pause_between_files = replayer_kwargs.pop(
        "pause_between_files",
        DEFAULT_PAUSE_BETWEEN_FILES,
    )
    try:
        if order == "random":
            file_iter = random_file_order(
                files,
                rng=rng,
                repeat=repeat,
                max_files=max_files,
            )
        elif order == "sequential":
            file_iter = sequential_file_order(
                files,
                repeat=repeat,
                max_files=max_files,
            )
        else:
            raise ValueError(f"order must be 'random' or 'sequential', got {order}")

        for parquet_file in file_iter:
            replayer.replay_file(parquet_file, **replayer_kwargs)
            if pause_between_files > 0:
                time.sleep(pause_between_files)
    finally:
        replayer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay qpos trajectories from LeRobot parquet files without "
            "starting the persistent joint impedance update loop."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--ip-address", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--order",
        choices=("random", "sequential"),
        default=DEFAULT_ORDER,
        help="Trajectory file order. random reshuffles every pass; sequential follows filename order.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Keep replaying trajectories until Ctrl+C.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum number of files to replay. 0 means all files once, or forever with --repeat.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Override source qpos rate. By default, timestamp deltas are used.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--reset-time", type=float, default=DEFAULT_RESET_TIME)
    parser.add_argument("--final-hold-time", type=float, default=0.2)
    parser.add_argument("--pause-between-files", type=float, default=DEFAULT_PAUSE_BETWEEN_FILES)
    parser.add_argument("--gain-scale", type=float, default=1.0)
    parser.add_argument("--max-joint-step", type=float, default=DEFAULT_MAX_JOINT_STEP)
    parser.add_argument("--allow-large-jumps", action="store_true")
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument(
        "--one-means-close",
        action="store_true",
        help="Use old gripper convention where qpos[7]=1 means close. Default is 1=open, 0=close.",
    )
    parser.add_argument("--open-width", type=float, default=DEFAULT_OPEN_WIDTH)
    return parser.parse_args()


def main():
    args = parse_args()
    replay_random_qpos_loop(
        data_dir=args.data_dir.expanduser(),
        pattern=args.pattern,
        order=args.order,
        repeat=args.repeat,
        max_files=args.max_files,
        seed=args.seed,
        ip_address=args.ip_address,
        port=args.port,
        stride=args.stride,
        control_hz=args.control_hz,
        reset_time=args.reset_time,
        final_hold_time=args.final_hold_time,
        pause_between_files=args.pause_between_files,
        gain_scale=args.gain_scale,
        max_joint_step=args.max_joint_step,
        allow_large_jumps=args.allow_large_jumps,
        disable_gripper=args.disable_gripper,
        one_means_close=args.one_means_close,
        open_width=args.open_width,
    )


if __name__ == "__main__":
    main()
