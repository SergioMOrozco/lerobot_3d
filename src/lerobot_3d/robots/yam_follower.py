"""One I2RT YAM arm as a LeRobot :class:`~lerobot.robots.robot.Robot`, read over ROS 2.

Joint states come from the shared :func:`~lerobot_3d.ros_hub.ros_hub` node, so any number of arms
(and the ROS cameras) share one subscription. Observations use LeRobot's ``"<motor>.pos"`` layout
(see :func:`~lerobot_3d.point_clouds.yam_robot_state.joint_keys`): radians for ``<side>_joint1..6``
and finger travel in metres for ``<side>_gripper``.

There is no YAM teleop yet, so the arm is observe-only: :meth:`YamFollower.send_action` raises.
The YAM's own controller holds joint calibration, so there is nothing to calibrate here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from lerobot_3d.point_clouds.yam_robot_state import joint_keys
from lerobot_3d.ros_hub import ros_hub, wait_until


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerConfig(RobotConfig):
    side: str = "left"
    """Which arm of the station: ``left`` or ``right`` (the URDF joint name prefix)."""
    joint_topic: str = "/joint_states"
    """``sensor_msgs/JointState`` topic carrying ``<side>_joint1..6`` (and optional ``joint7``)."""
    connect_timeout_s: float = 5.0
    """How long :meth:`YamFollower.connect` waits for this arm's first joint state."""

    def __post_init__(self):
        super().__post_init__()
        if self.side not in ("left", "right"):
            raise ValueError(f"YAM side must be 'left' or 'right', got {self.side!r}")


class YamFollower(Robot):
    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig):
        super().__init__(config)
        self.config = config
        self.keys = joint_keys(config.side)
        self._hub = None

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
        raise NotImplementedError(
            "The YAM is observe-only in lerobot_3d: there is no YAM teleop yet, so nothing commands it."
        )

    def disconnect(self) -> None:
        self._hub = None
