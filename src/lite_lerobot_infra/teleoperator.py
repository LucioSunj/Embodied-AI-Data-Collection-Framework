from __future__ import annotations

import abc

from .types import FeatureSpec, RobotAction, RobotObservation, TeleopEpisodeEvents


class Teleoperator(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    @abc.abstractmethod
    def action_features(self) -> FeatureSpec:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    def prime(self, observation: RobotObservation) -> None:
        return None

    @abc.abstractmethod
    def get_action(self, observation: RobotObservation) -> RobotAction:
        raise NotImplementedError

    def get_episode_events(self) -> TeleopEpisodeEvents:
        return TeleopEpisodeEvents()

    @abc.abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError
