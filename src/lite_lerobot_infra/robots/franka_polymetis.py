from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..camera import Camera
from ..robot import Robot
from ..types import FeatureSpec, RobotAction, RobotObservation

if TYPE_CHECKING:
    import grpc


JOINT_COUNT = 7


@dataclass(slots=True)
class WorkspaceBounds:
    x: tuple[float, float] = (0.25, 0.75)
    y: tuple[float, float] = (-0.45, 0.45)
    z: tuple[float, float] = (0.10, 0.65)

    def clip(self, position: np.ndarray) -> np.ndarray:
        clipped = np.asarray(position, dtype=np.float32).copy()
        clipped[0] = np.clip(clipped[0], *self.x)
        clipped[1] = np.clip(clipped[1], *self.y)
        clipped[2] = np.clip(clipped[2], *self.z)
        return clipped


@dataclass
class FrankaPolymetisConfig:
    ip_address: str = "localhost"
    robot_port: int = 50051
    gripper_port: int = 50052
    workspace_bounds: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    max_target_stretch_m: float = 0.03
    use_joint_velocities: bool = True
    use_ee_pose: bool = True
    use_gripper_state: bool = True
    cartesian_stiffness: tuple[float, float, float, float, float, float] | None = (
        200.0,
        200.0,
        200.0,
        20.0,
        20.0,
        20.0,
    )
    cartesian_damping: tuple[float, float, float, float, float, float] | None = None
    gripper_max_width: float = 0.08
    close_width_threshold: float = 0.005
    gripper_speed: float = 0.1
    gripper_force: float = 10.0
    gripper_change_threshold: float = 0.002
    cameras: dict[str, Camera] = field(default_factory=dict)


class FrankaPolymetisRobot(Robot):
    def __init__(self, config: FrankaPolymetisConfig):
        super().__init__(name="franka_polymetis")
        self.config = config
        self.cameras = config.cameras
        self._robot = None
        self._gripper = None
        self._grpc = None
        self._last_gripper_width = config.gripper_max_width

    @property
    def observation_features(self) -> FeatureSpec:
        features: FeatureSpec = OrderedDict()
        for idx in range(JOINT_COUNT):
            features[f"joint_{idx + 1}.pos"] = float
        if self.config.use_joint_velocities:
            for idx in range(JOINT_COUNT):
                features[f"joint_{idx + 1}.vel"] = float
        if self.config.use_ee_pose:
            for key in ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"):
                features[key] = float
        if self.config.use_gripper_state:
            features["gripper.width"] = float
            features["gripper.is_grasped"] = float
        for name, camera in self.cameras.items():
            features[name] = camera.frame_shape
        return features

    @property
    def action_features(self) -> FeatureSpec:
        return OrderedDict(
            {
                "ee.x": float,
                "ee.y": float,
                "ee.z": float,
                "ee.qx": float,
                "ee.qy": float,
                "ee.qz": float,
                "ee.qw": float,
                "gripper.width": float,
            }
        )

    @property
    def is_connected(self) -> bool:
        return self._robot is not None

    def connect(self) -> None:
        import grpc
        from polymetis import GripperInterface, RobotInterface

        self._grpc = grpc
        self._robot = RobotInterface(ip_address=self.config.ip_address, port=self.config.robot_port)

        try:
            self._gripper = GripperInterface(ip_address=self.config.ip_address, port=self.config.gripper_port)
            self._last_gripper_width = float(self._gripper.get_state().width)
        except grpc.RpcError:
            self._gripper = None
            self._last_gripper_width = self.config.gripper_max_width

        for camera in self.cameras.values():
            camera.connect()

        self._start_cartesian_controller()
        time.sleep(1.0)

    def _start_cartesian_controller(self) -> None:
        assert self._robot is not None
        stiffness = None
        damping = None
        if self.config.cartesian_stiffness is not None:
            stiffness = torch.tensor(self.config.cartesian_stiffness, dtype=torch.float32)
        if self.config.cartesian_damping is not None:
            damping = torch.tensor(self.config.cartesian_damping, dtype=torch.float32)
        self._robot.start_cartesian_impedance(Kx=stiffness, Kxd=damping)

    def _normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            _, current_quat = self._robot.get_ee_pose()
            return current_quat.detach().cpu().numpy().astype(np.float32)
        return quat / norm

    def _clip_pose_target(self, target_pos: np.ndarray) -> np.ndarray:
        assert self._robot is not None
        target_pos = self.config.workspace_bounds.clip(target_pos)
        current_pos, _ = self._robot.get_ee_pose()
        current_pos_np = current_pos.detach().cpu().numpy()
        delta = target_pos - current_pos_np
        distance = np.linalg.norm(delta)
        if distance > self.config.max_target_stretch_m and distance > 0:
            target_pos = current_pos_np + delta / distance * self.config.max_target_stretch_m
        return target_pos.astype(np.float32)

    def _send_gripper_width(self, target_width: float) -> None:
        if self._gripper is None:
            return

        if abs(target_width - self._last_gripper_width) < self.config.gripper_change_threshold:
            return

        try:
            if target_width <= self.config.close_width_threshold:
                self._gripper.grasp(
                    speed=self.config.gripper_speed,
                    force=self.config.gripper_force,
                    grasp_width=target_width,
                    blocking=False,
                )
            else:
                self._gripper.goto(
                    width=target_width,
                    speed=self.config.gripper_speed,
                    force=self.config.gripper_force,
                    blocking=False,
                )
            self._last_gripper_width = target_width
        except Exception:
            return

    def _recover_controller(self) -> RobotAction:
        assert self._robot is not None
        current_pos, current_quat = self._robot.get_ee_pose()
        target_pos = current_pos.detach().cpu().numpy().astype(np.float32)
        target_quat = current_quat.detach().cpu().numpy().astype(np.float32)
        self._start_cartesian_controller()
        return {
            "ee.x": float(target_pos[0]),
            "ee.y": float(target_pos[1]),
            "ee.z": float(target_pos[2]),
            "ee.qx": float(target_quat[0]),
            "ee.qy": float(target_quat[1]),
            "ee.qz": float(target_quat[2]),
            "ee.qw": float(target_quat[3]),
            "gripper.width": float(self._last_gripper_width),
        }

    def get_observation(self) -> RobotObservation:
        if self._robot is None:
            raise RuntimeError("Robot is not connected.")

        state = self._robot.get_robot_state()
        obs: RobotObservation = OrderedDict()

        for idx, value in enumerate(state.joint_positions):
            obs[f"joint_{idx + 1}.pos"] = float(value)

        if self.config.use_joint_velocities:
            for idx, value in enumerate(state.joint_velocities):
                obs[f"joint_{idx + 1}.vel"] = float(value)

        if self.config.use_ee_pose:
            ee_pos, ee_quat = self._robot.get_ee_pose()
            ee_pos_np = ee_pos.detach().cpu().numpy()
            ee_quat_np = ee_quat.detach().cpu().numpy()
            obs["ee.x"] = float(ee_pos_np[0])
            obs["ee.y"] = float(ee_pos_np[1])
            obs["ee.z"] = float(ee_pos_np[2])
            obs["ee.qx"] = float(ee_quat_np[0])
            obs["ee.qy"] = float(ee_quat_np[1])
            obs["ee.qz"] = float(ee_quat_np[2])
            obs["ee.qw"] = float(ee_quat_np[3])

        if self.config.use_gripper_state:
            if self._gripper is not None:
                try:
                    gripper_state = self._gripper.get_state()
                    obs["gripper.width"] = float(gripper_state.width)
                    obs["gripper.is_grasped"] = float(gripper_state.is_grasped)
                except self._grpc.RpcError:
                    obs["gripper.width"] = float(self._last_gripper_width)
                    obs["gripper.is_grasped"] = 0.0
            else:
                obs["gripper.width"] = float(self._last_gripper_width)
                obs["gripper.is_grasped"] = 0.0

        for name, camera in self.cameras.items():
            obs[name] = camera.read()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._robot is None:
            raise RuntimeError("Robot is not connected.")

        target_pos = np.asarray([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=np.float32)
        target_quat = np.asarray(
            [action["ee.qx"], action["ee.qy"], action["ee.qz"], action["ee.qw"]],
            dtype=np.float32,
        )
        target_width = float(action["gripper.width"])

        target_pos = self._clip_pose_target(target_pos)
        target_quat = self._normalize_quaternion(target_quat)

        try:
            self._robot.update_desired_ee_pose(
                position=torch.tensor(target_pos, dtype=torch.float32),
                orientation=torch.tensor(target_quat, dtype=torch.float32),
            )
            self._send_gripper_width(target_width)
        except self._grpc.RpcError:
            return self._recover_controller()

        return {
            "ee.x": float(target_pos[0]),
            "ee.y": float(target_pos[1]),
            "ee.z": float(target_pos[2]),
            "ee.qx": float(target_quat[0]),
            "ee.qy": float(target_quat[1]),
            "ee.qz": float(target_quat[2]),
            "ee.qw": float(target_quat[3]),
            "gripper.width": target_width,
        }

    def handle_aux_events(self, events: dict[str, object]) -> None:
        if not events:
            return
        if events.get("recover_gripper") and self._gripper is not None:
            try:
                from polymetis import GripperInterface

                self._gripper = GripperInterface(
                    ip_address=self.config.ip_address,
                    port=self.config.gripper_port,
                )
                if hasattr(self._gripper, "homing"):
                    self._gripper.homing()
            except Exception:
                return

    def disconnect(self) -> None:
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()

        if self._robot is not None:
            try:
                self._robot.terminate_current_policy(return_log=False)
            except Exception:
                pass
            self._robot = None

        self._gripper = None
        self._grpc = None
