"""One I2RT YAM arm as a LeRobot :class:`~lerobot.robots.robot.Robot`, over ROS 2.

Joint states come from the arm's topic (default ``/<side>/joint_states``) through the shared
:func:`~lerobot_3d.ros_hub.ros_hub` node, which the other arm and the ROS cameras share. Observations
and actions use LeRobot's ``"<motor>.pos"`` layout (see
:func:`~lerobot_3d.point_clouds.yam_robot_state.joint_keys`): radians for ``<side>_joint1..6`` and the
opening, 0 closed .. 1 open, for ``<side>_gripper``.

:meth:`YamFollower.send_action` publishes a ``sensor_msgs/JointState`` on the arm's command topic
(default ``/<side>/cmd_joint``). The YAM's controller rejects targets more than 0.35 rad from the
current position, so each command is limited twice: the target moves at most
``max_joint_speed_rad_s`` from the previous command, and stays within ``max_step_rad`` of the measured
joints. The YAM's own controller holds joint calibration, so there is nothing to calibrate here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from lerobot_3d.point_clouds.yam_robot_state import joint_keys
from lerobot_3d.ros_hub import ros_hub, wait_until


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerConfig(RobotConfig):
    side: str = "left"
    """Which arm of the station: ``left`` or ``right`` (the URDF joint name prefix)."""
    joint_topic: str | None = None
    """``sensor_msgs/JointState`` with ``<side>_joint1..6`` and ``<side>_gripper``; ``None`` ->
    ``/<side>/joint_states``."""
    command_topic: str | None = None
    """Where :meth:`YamFollower.send_action` publishes ``JointState`` targets; ``None`` -> ``/<side>/cmd_joint``."""
    connect_timeout_s: float = 5.0
    """How long :meth:`YamFollower.connect` waits for this arm's first joint state."""
    max_joint_speed_rad_s: float = 1.0
    """Fastest a joint target may move between commands (radians per second)."""
    max_gripper_speed: float = 1.5
    """Fastest the gripper target may move (full openings per second)."""
    max_step_rad: float = 0.3
    """Hard cap on how far a target may be from the measured joint (the YAM rejects > 0.35 rad)."""

    def __post_init__(self):
        super().__post_init__()
        if self.side not in ("left", "right"):
            raise ValueError(f"YAM side must be 'left' or 'right', got {self.side!r}")
        if self.joint_topic is None:
            self.joint_topic = f"/{self.side}/joint_states"
        if self.command_topic is None:
            self.command_topic = f"/{self.side}/cmd_joint"


class YamFollower(Robot):
    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig):
        super().__init__(config)
        self.config = config
        self.keys = joint_keys(config.side)
        self.names = [key.removesuffix(".pos") for key in self.keys]
        """ROS joint names, in key order: ``<side>_joint1..6``, ``<side>_gripper``."""
        self._hub = None
        self._last_command: np.ndarray | None = None
        self._last_command_time = 0.0

    @property
    def observation_features(self) -> dict:
        return dict.fromkeys(self.keys, float)

    @property
    def action_features(self) -> dict:
        return dict.fromkeys(self.keys, float)

    @property
    def is_connected(self) -> bool:
        return self._hub is not None

    def connect(self, calibrate: bool = True) -> None:
        hub = ros_hub()
        hub.watch_joints(self.config.joint_topic)
        if not wait_until(lambda: hub.latest_arm(self.config.joint_topic, self.config.side),
                          self.config.connect_timeout_s):
            raise ConnectionError(
                f"No {self.config.side}_joint1..6 on {self.config.joint_topic} within "
                f"{self.config.connect_timeout_s:.0f} s; is the YAM's ROS driver running?"
            )
        self._hub = hub

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        """Latest measured joints; ``observation_age_s`` on the robot says how old they are."""
        if self._hub is None:
            raise ConnectionError(f"{self} is not connected.")
        sample = self._hub.latest_arm(self.config.joint_topic, self.config.side)
        self.observation_age_s = time.monotonic() - sample.arrival
        return dict(sample.observation)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Publish ``action``'s joint targets, speed- and step-limited; returns what was actually sent."""
        if self._hub is None:
            raise ConnectionError(f"{self} is not connected.")
        try:
            target = np.array([float(action[key]) for key in self.keys])
        except KeyError as e:
            raise ValueError(f"YAM {self.config.side} action is missing {e}") from e
        measured = np.array([self.get_observation()[key] for key in self.keys])
        now = time.monotonic()
        previous = measured if self._last_command is None else self._last_command
        dt = min(now - self._last_command_time, 0.1) if self._last_command is not None else 0.0
        speed = np.array([self.config.max_joint_speed_rad_s] * 6 + [self.config.max_gripper_speed])
        command = np.clip(target, previous - speed * dt, previous + speed * dt)
        step = self.config.max_step_rad
        command[:6] = np.clip(command[:6], measured[:6] - step, measured[:6] + step)
        command[6] = np.clip(command[6], 0.0, 1.0)
        self._hub.publish_joint_command(self.config.command_topic, self.names, command)
        self._last_command, self._last_command_time = command, now
        return {key: float(v) for key, v in zip(self.keys, command)}

    def disconnect(self) -> None:
        self._hub = None
        self._last_command = None
