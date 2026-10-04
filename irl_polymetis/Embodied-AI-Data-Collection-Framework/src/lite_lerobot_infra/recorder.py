from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from lerobot.datasets import LeRobotDataset

from .features import ACTION, OBSERVATION, build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from .robot import Robot
from .teleoperator import Teleoperator


@dataclass(slots=True)
class DatasetRecorderConfig:
    repo_id: str
    task: str
    root: str | Path
    fps: int = 30
    use_videos: bool = True
    start_recording_immediately: bool = False
    resume: bool = False
    save_on_exit: bool = True
    max_episodes: int | None = None


class DatasetRecorder:
    def __init__(self, robot: Robot, teleop: Teleoperator, config: DatasetRecorderConfig):
        self.robot = robot
        self.teleop = teleop
        self.config = config
        self.dataset: LeRobotDataset | None = None

    def _validate_action_schema(self) -> None:
        if self.teleop.action_features != self.robot.action_features:
            raise ValueError(
                "teleop.action_features must exactly match robot.action_features so the recorded "
                "dataset action can be replayed back into the robot."
            )

    def _dataset_features(self) -> dict[str, dict]:
        return combine_feature_dicts(
            hw_to_dataset_features(self.robot.action_features, ACTION, use_videos=self.config.use_videos),
            hw_to_dataset_features(self.robot.observation_features, OBSERVATION, use_videos=self.config.use_videos),
        )

    def _create_dataset(self) -> LeRobotDataset:
        features = self._dataset_features()
        root = Path(self.config.root)
        if self.config.resume:
            return LeRobotDataset.resume(
                self.config.repo_id,
                root=root,
            )
        return LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            root=root,
            robot_type=self.robot.name,
            features=features,
            use_videos=self.config.use_videos,
        )

    def run(self) -> LeRobotDataset:
        self._validate_action_schema()
        self.dataset = self._create_dataset()

        try:
            self.robot.connect()

            initial_obs = self.robot.get_observation()
            self.teleop.connect()
            self.teleop.prime(initial_obs)

            recording = self.config.start_recording_immediately
            recorded_episodes = 0

            while True:
                loop_start = time.perf_counter()

                obs = self.robot.get_observation()
                events = self.teleop.get_episode_events()
                self.robot.handle_aux_events(events.metadata)

                if events.start_episode and not recording:
                    if self.dataset.has_pending_frames():
                        self.dataset.clear_episode_buffer()
                    self.teleop.prime(obs)
                    recording = True

                action = self.teleop.get_action(obs)
                sent_action = self.robot.send_action(action)

                if recording:
                    observation_frame = build_dataset_frame(self.dataset.features, obs, OBSERVATION)
                    action_frame = build_dataset_frame(self.dataset.features, sent_action, ACTION)
                    self.dataset.add_frame({**observation_frame, **action_frame, "task": self.config.task})

                if events.rerecord_episode and recording:
                    self.dataset.clear_episode_buffer()
                    recording = False

                if events.stop_episode and recording:
                    if self.dataset.has_pending_frames():
                        self.dataset.save_episode()
                        recorded_episodes += 1
                    recording = False

                if events.exit_requested:
                    if recording and self.config.save_on_exit and self.dataset.has_pending_frames():
                        self.dataset.save_episode()
                    elif recording and self.dataset.has_pending_frames():
                        self.dataset.clear_episode_buffer()
                    break

                if self.config.max_episodes is not None and recorded_episodes >= self.config.max_episodes:
                    break

                dt = time.perf_counter() - loop_start
                time.sleep(max(1.0 / self.config.fps - dt, 0.0))
        finally:
            if self.teleop.is_connected:
                self.teleop.disconnect()
            if self.robot.is_connected:
                self.robot.disconnect()
            if self.dataset is not None:
                self.dataset.finalize()

        return self.dataset
