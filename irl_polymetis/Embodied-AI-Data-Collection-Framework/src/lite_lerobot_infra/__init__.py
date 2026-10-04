from .camera import Camera
from .robot import Robot
from .teleoperator import Teleoperator
from .types import RobotAction, RobotObservation, TeleopEpisodeEvents

__all__ = [
    "Camera",
    "DatasetRecorder",
    "DatasetRecorderConfig",
    "Robot",
    "RobotAction",
    "RobotObservation",
    "Teleoperator",
    "TeleopEpisodeEvents",
]


def __getattr__(name: str):
    if name in {"DatasetRecorder", "DatasetRecorderConfig"}:
        from .recorder import DatasetRecorder, DatasetRecorderConfig

        exports = {
            "DatasetRecorder": DatasetRecorder,
            "DatasetRecorderConfig": DatasetRecorderConfig,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
