from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

RobotAction: TypeAlias = dict[str, Any]
RobotObservation: TypeAlias = dict[str, Any]
FeatureSpec: TypeAlias = dict[str, type | tuple[int, ...]]


@dataclass(slots=True)
class TeleopEpisodeEvents:
    start_episode: bool = False
    stop_episode: bool = False
    rerecord_episode: bool = False
    exit_requested: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
