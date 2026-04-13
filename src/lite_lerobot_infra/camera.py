from __future__ import annotations

import abc

import numpy as np


class Camera(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def frame_shape(self) -> tuple[int, int, int]:
        raise NotImplementedError

    @abc.abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def read(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError
