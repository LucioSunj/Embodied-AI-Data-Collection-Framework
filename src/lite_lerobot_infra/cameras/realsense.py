from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ..camera import Camera


@dataclass(slots=True)
class RealSenseCameraConfig:
    name: str
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 10
    hardware_reset: bool = False
    disable_emitter: bool = True


class RealSenseCamera(Camera):
    def __init__(self, config: RealSenseCameraConfig):
        super().__init__(config.name)
        self.config = config
        self._pipeline = None
        self._rs = None
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None and self._reader_thread is not None and self._reader_thread.is_alive()

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (self.config.height, self.config.width, 3)

    def connect(self) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        if self.config.hardware_reset:
            ctx = rs.context()
            for device in ctx.query_devices():
                serial = device.get_info(rs.camera_info.serial_number)
                if serial == self.config.serial:
                    device.hardware_reset()
                    time.sleep(3.0)
                    break

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.config.serial)
        config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )

        profile = pipeline.start(config)
        device = profile.get_device()
        if self.config.disable_emitter:
            for sensor in device.query_sensors():
                if sensor.is_depth_sensor() and sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(rs.option.emitter_enabled, 0)

        for _ in range(self.config.warmup_frames):
            pipeline.wait_for_frames(timeout_ms=2000)

        self._pipeline = pipeline
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._reader_thread.start()

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data()).copy()
                with self._frame_lock:
                    self._latest_frame = image
            except Exception:
                time.sleep(0.01)

    def read(self) -> np.ndarray:
        with self._frame_lock:
            if self._latest_frame is None:
                raise RuntimeError(f"Camera '{self.name}' has no frame available yet.")
            return self._latest_frame.copy()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
