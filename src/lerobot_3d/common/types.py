"""Shared data types referenced across the point-cloud/teleop pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Datapoint:
    """One camera's frame: raw color/depth plus what's needed to fuse it into world frame."""

    serial: str
    color: np.ndarray | None
    depth: np.ndarray
    depth_scale: float
    max_depth: float
    X_WC: np.ndarray
    color_intrinsics: object  # pyrealsense2.intrinsics-like: needs .fx/.fy/.ppx/.ppy
    obj_mask: np.ndarray | None = None
    joint_positions: dict[str, float] | None = None
    """Raw motor-space observation (``"<joint>.pos"`` keys) from the follower that produced
    this step's robot state -- same format as ``Follower.get_observation()``/``send_action()``,
    so it can be fed straight into ``RobotState`` or a new action without reconversion."""


@dataclass
class RobotSnapshot:
    """One robot's state for a single step, in that robot's own base frame.

    Identical joint positions give identical ``pcd``/``link_pcds``/``link_poses`` regardless of
    where the robot is drawn; ``base_offset`` is only where viser places it on the grid.
    """

    index: int
    joint_positions: dict[str, float]
    """Motor-space ``"<joint>.pos"`` values (the action, or the follower's observation)."""
    joint_radians: dict[str, float]
    """Calibrated joint angles (radians) used for FK."""
    pcd: np.ndarray
    """``(M, 3)`` ``float64`` sampled robot mesh points, robot base frame."""
    link_pcds: dict[str, np.ndarray]
    """Per-link sampled points keyed by URDF link name, robot base frame."""
    link_poses: dict[str, tuple[np.ndarray, np.ndarray]]
    """Per-link ``(translation, quaternion_wxyz)``, robot base frame."""
    base_offset: np.ndarray
    """``(3,)`` position of this robot's base in the viser world (grid layout)."""
    base_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """World orientation of the base; identity preserves existing SO101 layouts."""
