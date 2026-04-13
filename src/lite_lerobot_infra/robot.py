from __future__ import annotations

import abc
from typing import Any

from .types import FeatureSpec, RobotAction, RobotObservation


class Robot(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    @abc.abstractmethod
    def observation_features(self) -> FeatureSpec:
        raise NotImplementedError

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

    @abc.abstractmethod
    def get_observation(self) -> RobotObservation:
        raise NotImplementedError

    @abc.abstractmethod
    def send_action(self, action: RobotAction) -> RobotAction:
        raise NotImplementedError

    def handle_aux_events(self, events: dict[str, Any]) -> None:
        return None

    @abc.abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError
