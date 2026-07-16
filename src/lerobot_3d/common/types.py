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
