from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..teleoperator import Teleoperator
from ..types import FeatureSpec, RobotAction, RobotObservation, TeleopEpisodeEvents


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    sign = np.sign(value)
    return float(sign * (abs(value) - deadzone) / (1.0 - deadzone))


@dataclass(slots=True)
class XboxTeleopConfig:
    deadzone: float = 0.15
    max_linear_velocity_mps: float = 0.06
    max_yaw_velocity_radps: float = 0.5
    control_dt: float = 0.01
    gripper_max_width: float = 0.08
    gripper_trigger_threshold: float = 0.5
    axis_left_x: int = 0
    axis_left_y: int = 1
    axis_right_x: int = 3
    axis_right_y: int = 4
    axis_right_trigger: int = 5
    button_start: int = 7
    button_back: int = 6
    button_stop_episode: int = 1
    button_rerecord_episode: int = 3


class XboxTeleoperator(Teleoperator):
    def __init__(self, config: XboxTeleopConfig):
        super().__init__(name="xbox")
        self.config = config
        self._pygame = None
        self._joystick = None
        self._target_pos: np.ndarray | None = None
        self._target_euler: np.ndarray | None = None
        self._previous_buttons: dict[int, bool] = {}

    @property
    def action_features(self) -> FeatureSpec:
        return {
            "ee.x": float,
            "ee.y": float,
            "ee.z": float,
            "ee.qx": float,
            "ee.qy": float,
            "ee.qz": float,
            "ee.qw": float,
            "gripper.width": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None

    def connect(self) -> None:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No Xbox-compatible controller detected.")

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        self._pygame = pygame
        self._joystick = joystick
        self._previous_buttons.clear()

    def prime(self, observation: RobotObservation) -> None:
        quat = np.asarray(
            [
                observation["ee.qx"],
                observation["ee.qy"],
                observation["ee.qz"],
                observation["ee.qw"],
            ],
            dtype=np.float32,
        )
        self._target_pos = np.asarray(
            [observation["ee.x"], observation["ee.y"], observation["ee.z"]],
            dtype=np.float32,
        )
        self._target_euler = R.from_quat(quat).as_euler("zyx", degrees=False).astype(np.float32)

    def _button(self, button_id: int) -> bool:
        assert self._joystick is not None
        return bool(self._joystick.get_button(button_id))

    def _button_edge(self, button_id: int) -> bool:
        current = self._button(button_id)
        previous = self._previous_buttons.get(button_id, False)
        self._previous_buttons[button_id] = current
        return current and not previous

    def _poll_inputs(self) -> dict[str, float]:
        assert self._pygame is not None
        assert self._joystick is not None
        self._pygame.event.pump()

        raw_lx = self._joystick.get_axis(self.config.axis_left_x)
        raw_ly = self._joystick.get_axis(self.config.axis_left_y)
        raw_rx = self._joystick.get_axis(self.config.axis_right_x)
        raw_ry = self._joystick.get_axis(self.config.axis_right_y)
        raw_rt = self._joystick.get_axis(self.config.axis_right_trigger)

        return {
            "vx": -apply_deadzone(raw_ly, self.config.deadzone),
            "vy": -apply_deadzone(raw_lx, self.config.deadzone),
            "vz": -apply_deadzone(raw_ry, self.config.deadzone),
            "yaw_rate": apply_deadzone(raw_rx, self.config.deadzone),
            "right_trigger": raw_rt,
        }

    def get_episode_events(self) -> TeleopEpisodeEvents:
        if not self.is_connected:
            return TeleopEpisodeEvents()

        self._poll_inputs()

        start_edge = self._button_edge(self.config.button_start)
        stop_edge = self._button_edge(self.config.button_stop_episode)
        rerecord_edge = self._button_edge(self.config.button_rerecord_episode)
        back_pressed = self._button(self.config.button_back)
        back_edge = self._button_edge(self.config.button_back)
        exit_requested = back_pressed and self._button(self.config.button_start)

        metadata: dict[str, object] = {}
        if back_edge and not exit_requested:
            metadata["recover_gripper"] = True

        return TeleopEpisodeEvents(
            start_episode=start_edge and not exit_requested,
            stop_episode=stop_edge and not exit_requested,
            rerecord_episode=rerecord_edge and not exit_requested,
            exit_requested=exit_requested,
            metadata=metadata,
        )

    def get_action(self, observation: RobotObservation) -> RobotAction:
        if self._target_pos is None or self._target_euler is None:
            self.prime(observation)

        state = self._poll_inputs()
        assert self._target_pos is not None
        assert self._target_euler is not None

        self._target_pos[0] += state["vx"] * self.config.max_linear_velocity_mps * self.config.control_dt
        self._target_pos[1] += state["vy"] * self.config.max_linear_velocity_mps * self.config.control_dt
        self._target_pos[2] += state["vz"] * self.config.max_linear_velocity_mps * self.config.control_dt
        self._target_euler[0] += state["yaw_rate"] * self.config.max_yaw_velocity_radps * self.config.control_dt

        normalized_rt = (state["right_trigger"] + 1.0) / 2.0
        gripper_width = 0.0 if normalized_rt > self.config.gripper_trigger_threshold else self.config.gripper_max_width
        quat = R.from_euler("zyx", self._target_euler).as_quat().astype(np.float32)

        return {
            "ee.x": float(self._target_pos[0]),
            "ee.y": float(self._target_pos[1]),
            "ee.z": float(self._target_pos[2]),
            "ee.qx": float(quat[0]),
            "ee.qy": float(quat[1]),
            "ee.qz": float(quat[2]),
            "ee.qw": float(quat[3]),
            "gripper.width": float(gripper_width),
        }

    def disconnect(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._joystick = None
