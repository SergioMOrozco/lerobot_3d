"""RGB-D cameras published over ROS 2, as the same :class:`Datapoint` objects ``MultiRealSenseStream``
produces -- color BGR, depth registered to the color image (mm), color intrinsics, and ``X_WC`` -- so
fusion, Capture and dataset recording treat ROS and RealSense cameras identically.

Each camera is named after its namespace (``/camera/top`` -> ``camera_top``, see :func:`camera_key`);
that name is its key in ``calibration_files/`` and in the intrinsic/extrinsic JSON files.

- **Fixed cameras** (no ``link``, e.g. an overhead camera) work exactly like RealSense ones: ``X_WC``
  comes from the camera's entry in the extrinsics file, calibrated with Capture + ``icp.py``.
- **Linked cameras** (``link`` set, e.g. a wrist camera) are placed at that URDF link's pose from the
  measured joints. Once calibrated, the extrinsics entry also holds the ``link`` and its pose at
  capture (``X_WL``), and the camera sits at ``link pose @ inv(X_WL) @ X_WC`` -- the calibrated mount.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot_3d.common.types import Datapoint
from lerobot_3d.paths import resolve_extrinsic_calibration_json
from lerobot_3d.point_clouds.rgbd import register_depth_to_color
from lerobot_3d.ros_hub import ros_hub, wait_until

DEPTH_SCALE_MM = 0.001
"""ROS ``16UC1`` depth is millimetres."""
MAX_DEPTH_M = 10.0
"""Depth truncation, same as ``MultiRealSenseStream``."""


def camera_key(namespace: str) -> str:
    """Camera namespace -> its name in ``calibration_files/`` and the calibration JSON files."""
    return namespace.strip("/").replace("/", "_")


def extrinsics_path(extrinsics_file: str | Path) -> Path:
    """The extrinsics file teleop reads (see ``resolve_extrinsic_calibration_json``), or
    ``extrinsics_file`` relative to the working directory if it doesn't exist yet."""
    try:
        return resolve_extrinsic_calibration_json(extrinsics_file)
    except FileNotFoundError:
        return Path(extrinsics_file)


def read_extrinsics(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def calibrated_mount(entry: Mapping | None, link: str) -> np.ndarray | None:
    """``X_link_cam`` from an extrinsics entry captured on ``link``, else ``None`` (nominal URDF mount)."""
    if not entry or entry.get("link") != link or "X_WL" not in entry:
        return None
    return np.linalg.inv(np.asarray(entry["X_WL"], dtype=np.float64)) @ np.asarray(entry["X_WC"], dtype=np.float64)


class RosCameraStream:
    """Latest frames from ROS cameras (``RosCameraConfig`` entries), one :class:`Datapoint` each."""

    def __init__(
        self,
        cameras: Sequence,
        extrinsics_file: str | Path,
        *,
        connect_timeout_s: float = 10.0,
        hub=None,
    ):
        self.cameras = list(cameras)
        self.keys = [camera_key(cam.namespace) for cam in self.cameras]
        self.extrinsics_file = extrinsics_path(extrinsics_file)
        extrinsics = read_extrinsics(self.extrinsics_file)
        self.X_WC: dict[str, np.ndarray] = {}
        """Fixed cameras' world poses."""
        self.mounts: dict[str, np.ndarray | None] = {}
        """Linked cameras' calibrated mounts (``None`` -> nominal URDF mount)."""
        for cam, key in zip(self.cameras, self.keys):
            if cam.link is None:
                if key not in extrinsics:
                    raise ValueError(
                        f"ROS camera {cam.namespace} has no '{key}' entry in {self.extrinsics_file}. Add one "
                        'with an identity "X_WC" to start, then calibrate it with Capture + icp.py.'
                    )
                self.X_WC[key] = np.asarray(extrinsics[key]["X_WC"], dtype=np.float64)
            else:
                self.mounts[key] = calibrated_mount(extrinsics.get(key), cam.link)
        self._registered: dict[str, tuple[tuple[int, int], np.ndarray]] = {}

        self._hub = ros_hub() if hub is None else hub
        for cam in self.cameras:
            self._hub.watch_camera(
                cam.namespace, color_topic=cam.color_topic, depth_topic=cam.depth_topic,
                color_info_topic=cam.color_info_topic, depth_info_topic=cam.depth_info_topic,
            )
        if not wait_until(self._all_ready, connect_timeout_s):
            missing = [cam.namespace for cam in self.cameras if not self._ready(cam.namespace)]
            raise ConnectionError(
                f"No complete RGB-D stream within {connect_timeout_s:.0f} s from {missing} (needs color, "
                "depth, both camera_info topics, and the depth->color transform on /tf_static)."
            )

    @property
    def links(self) -> set[str]:
        """URDF links whose world pose :meth:`get_datapoints` needs."""
        return {cam.link for cam in self.cameras if cam.link is not None}

    def _ready(self, namespace) -> bool:
        frame = self._hub.latest_camera(namespace)
        return frame is not None and frame.ready

    def _all_ready(self) -> bool:
        return all(self._ready(cam.namespace) for cam in self.cameras)

    def _depth_on_color(self, key, frame) -> np.ndarray:
        # Registration is the expensive part; do it once per new depth/color frame pair.
        stamp = (frame.depth_count, frame.color_count)
        cached = self._registered.get(key)
        if cached is None or cached[0] != stamp:
            depth = register_depth_to_color(
                frame.depth, frame.K_depth, DEPTH_SCALE_MM, frame.T_color_depth, frame.K_color, frame.color.shape
            )
            cached = self._registered[key] = (stamp, depth)
        return cached[1]

    def camera_pose(self, key: str, link: str | None, link_poses: Mapping[str, np.ndarray]) -> np.ndarray:
        if link is None:
            return self.X_WC[key]
        mount = self.mounts[key]
        return link_poses[link] if mount is None else link_poses[link] @ mount

    def get_datapoints(self, link_poses: Mapping[str, np.ndarray] | None = None) -> list[Datapoint]:
        """One datapoint per camera; ``link_poses`` maps each of :attr:`links` to its world pose."""
        link_poses = link_poses or {}
        datapoints = []
        for cam, key in zip(self.cameras, self.keys):
            frame = self._hub.latest_camera(cam.namespace)
            K = frame.K_color
            datapoints.append(Datapoint(
                serial=key,
                color=np.ascontiguousarray(frame.color[:, :, ::-1]),  # RGB -> BGR, like RealSense bgr8
                depth=self._depth_on_color(key, frame),
                depth_scale=DEPTH_SCALE_MM,
                max_depth=MAX_DEPTH_M,
                X_WC=self.camera_pose(key, cam.link, link_poses),
                color_intrinsics=SimpleNamespace(fx=float(K[0, 0]), fy=float(K[1, 1]),
                                                 ppx=float(K[0, 2]), ppy=float(K[1, 2])),
            ))
        return datapoints

    def extrinsic_seeds(self, link_poses: Mapping[str, np.ndarray]) -> dict[str, dict]:
        """Extrinsics entries for linked cameras at the current joints, written on Capture: the current
        placement as ``icp.py``'s starting ``X_WC``, plus ``link`` and ``X_WL`` so the result becomes a mount."""
        seeds = {}
        for cam, key in zip(self.cameras, self.keys):
            if cam.link is not None:
                seeds[key] = {
                    "X_WC": self.camera_pose(key, cam.link, link_poses).tolist(),
                    "link": cam.link,
                    "X_WL": np.asarray(link_poses[cam.link]).tolist(),
                }
        return seeds

    def stop(self) -> None:
        """Nothing to release per stream; the shared ROS node is shut down by its owner."""
