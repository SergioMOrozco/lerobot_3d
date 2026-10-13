"""Record teleop steps into a ``LeRobotDataset``: RGB-D, camera parameters, actions, joint
state, and per-link robot point clouds.

Per-frame schema (``<s>`` = camera serial, ``<link>`` = URDF link name):

- ``action`` / ``observation.state``: float32 ``(J,)`` motor-space joint values.
- ``observation.images.<s>``: RGB video.
- ``observation.depth.<s>``: uint16 depth packed losslessly into an RGB PNG image
  (see :mod:`lerobot_3d.recording.depth_codec`).
- ``observation.depth_scale.<s>``: float32 ``(1,)`` meters per depth unit.
- ``observation.intrinsics.<s>``: float32 ``(3, 3)`` color camera K (depth is aligned to color).
- ``observation.extrinsics.<s>``: float32 ``(4, 4)`` ``X_WC`` (camera -> world).
- ``observation.robot_link_pcds.<link>``: float32 ``(N, 3)`` world-frame points; point ``k``
  of a link is the same body point in every frame.

With several robots (e.g. both YAM arms) their joints and links are merged into the same features, in
robot order; names must be unique across robots (YAM's are prefixed with the arm's side).

Use :mod:`lerobot_3d.recording.dataset_loader` to turn a loaded frame back into
:class:`~lerobot_3d.common.types.Datapoint` objects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from scipy.spatial.transform import Rotation

from lerobot_3d.common.types import Datapoint, RobotSnapshot
from lerobot_3d.recording.depth_codec import DEPTH_ENCODING, pack_depth

SIDECAR_NAME = "lerobot_3d.json"
"""Extra metadata written to ``<root>/meta/`` next to LeRobot's own ``info.json``."""
SIDECAR_FORMAT_VERSION = 1

ACTION_KEY = "action"
STATE_KEY = "observation.state"
RGB_PREFIX = "observation.images."
DEPTH_PREFIX = "observation.depth."
DEPTH_SCALE_PREFIX = "observation.depth_scale."
INTRINSICS_PREFIX = "observation.intrinsics."
EXTRINSICS_PREFIX = "observation.extrinsics."
LINK_PCD_PREFIX = "observation.robot_link_pcds."

_IMAGE_NAMES = ["height", "width", "channels"]


def intrinsics_matrix(color_intrinsics) -> np.ndarray:
    """``(3, 3)`` K from a pyrealsense2-like intrinsics object (``.fx/.fy/.ppx/.ppy``)."""
    return np.array(
        [
            [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
            [0.0, color_intrinsics.fy, color_intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _per_robot(snapshots, actions) -> tuple[list[RobotSnapshot], list[dict[str, float]]]:
    """Accept one robot's ``(snapshot, action)`` or per-robot lists of them."""
    if isinstance(snapshots, RobotSnapshot):
        return [snapshots], [actions]
    return list(snapshots), list(actions)


def _merged(dicts: Sequence[dict], what: str) -> dict:
    merged = {}
    for d in dicts:
        clash = merged.keys() & d.keys()
        if clash:
            raise ValueError(f"Duplicate {what} across robots: {sorted(clash)}")
        merged.update(d)
    return merged


def world_link_pcds(snapshots: Sequence[RobotSnapshot]) -> dict[str, np.ndarray]:
    """Every robot's link clouds moved from its base frame into the world frame, merged."""
    per_robot = []
    for snapshot in snapshots:
        offset = np.asarray(snapshot.base_offset, dtype=np.float64)
        if tuple(snapshot.base_wxyz) == (1.0, 0.0, 0.0, 0.0) and not offset.any():
            per_robot.append(dict(snapshot.link_pcds))  # base frame is the world frame
            continue
        w, x, y, z = snapshot.base_wxyz
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
        per_robot.append({
            name: np.asarray(pts, dtype=np.float64) @ R.T + offset for name, pts in snapshot.link_pcds.items()
        })
    return _merged(per_robot, "link names")


def build_features(
    datapoints: Sequence[Datapoint],
    snapshots: Sequence[RobotSnapshot] | RobotSnapshot,
    actions: Sequence[dict[str, float]] | dict[str, float],
) -> dict:
    """LeRobot feature spec for one step's data (shapes come from the real frames)."""
    snapshots, actions = _per_robot(snapshots, actions)
    action_names = list(_merged(actions, "action names"))
    state_names = list(_merged([s.joint_positions for s in snapshots], "joint names"))
    features: dict = {
        ACTION_KEY: {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        STATE_KEY: {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
    }
    for dp in datapoints:
        h, w = dp.depth.shape[:2]
        s = dp.serial
        features[f"{RGB_PREFIX}{s}"] = {"dtype": "video", "shape": (h, w, 3), "names": _IMAGE_NAMES}
        features[f"{DEPTH_PREFIX}{s}"] = {"dtype": "image", "shape": (h, w, 3), "names": _IMAGE_NAMES}
        features[f"{DEPTH_SCALE_PREFIX}{s}"] = {"dtype": "float32", "shape": (1,), "names": None}
        features[f"{INTRINSICS_PREFIX}{s}"] = {"dtype": "float32", "shape": (3, 3), "names": None}
        features[f"{EXTRINSICS_PREFIX}{s}"] = {"dtype": "float32", "shape": (4, 4), "names": None}
    for link_name, pts in world_link_pcds(snapshots).items():
        features[f"{LINK_PCD_PREFIX}{link_name}"] = {
            "dtype": "float32",
            "shape": tuple(np.asarray(pts).shape),
            "names": None,
        }
    return features


def build_frame(
    datapoints: Sequence[Datapoint],
    snapshots: Sequence[RobotSnapshot] | RobotSnapshot,
    actions: Sequence[dict[str, float]] | dict[str, float],
    features: dict,
    task: str,
) -> dict:
    """One ``LeRobotDataset.add_frame`` dict, laid out per :func:`build_features`."""
    snapshots, actions = _per_robot(snapshots, actions)
    action = _merged(actions, "action names")
    joint_positions = _merged([s.joint_positions for s in snapshots], "joint names")
    action_names = features[ACTION_KEY]["names"]
    state_names = features[STATE_KEY]["names"]
    frame = {
        ACTION_KEY: np.array([action[k] for k in action_names], dtype=np.float32),
        STATE_KEY: np.array([joint_positions[k] for k in state_names], dtype=np.float32),
        "task": task,
    }
    for dp in datapoints:
        s = dp.serial
        # RealSense streams bgr8; LeRobot videos are RGB.
        frame[f"{RGB_PREFIX}{s}"] = cv2.cvtColor(np.asarray(dp.color), cv2.COLOR_BGR2RGB)
        frame[f"{DEPTH_PREFIX}{s}"] = pack_depth(np.asarray(dp.depth, dtype=np.uint16))
        frame[f"{DEPTH_SCALE_PREFIX}{s}"] = np.array([dp.depth_scale], dtype=np.float32)
        frame[f"{INTRINSICS_PREFIX}{s}"] = intrinsics_matrix(dp.color_intrinsics)
        frame[f"{EXTRINSICS_PREFIX}{s}"] = np.asarray(dp.X_WC, dtype=np.float32)
    for link_name, pts in world_link_pcds(snapshots).items():
        frame[f"{LINK_PCD_PREFIX}{link_name}"] = np.asarray(pts, dtype=np.float32)
    return frame


def _comparable(features: dict) -> dict:
    """Only the keys this module writes, with JSON-roundtripped shapes normalized to tuples."""
    prefixes = (RGB_PREFIX, DEPTH_PREFIX, DEPTH_SCALE_PREFIX, INTRINSICS_PREFIX,
                EXTRINSICS_PREFIX, LINK_PCD_PREFIX)
    return {
        key: (ft["dtype"], tuple(ft["shape"]))
        for key, ft in features.items()
        if key in (ACTION_KEY, STATE_KEY) or key.startswith(prefixes)
    }


class LeRobotDatasetRecorder:
    """Append teleop episodes to a local ``LeRobotDataset``.

    The dataset is created on the first recorded frame (its shapes come from real data), or
    opened for appending if ``root`` already holds one. Call :meth:`finalize` when done --
    LeRobot datasets are not loadable until their parquet writers are closed.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | Path | None = None,
        fps: int = 15,
        task: str = "teleop",
        robot_type: str = "so101_follower",
        image_writer_threads_per_camera: int = 4,
    ):
        self.repo_id = repo_id
        self.root = Path(root).expanduser() if root is not None else HF_LEROBOT_HOME / repo_id
        self.fps = int(fps)
        self.task = task
        self.robot_type = robot_type
        self.image_writer_threads_per_camera = image_writer_threads_per_camera

        self.dataset: LeRobotDataset | None = None
        self.features: dict | None = None
        self.recording = False
        self.episode_frames = 0

        if (self.root / "meta" / "info.json").is_file():
            self.dataset = LeRobotDataset(repo_id, root=self.root)
            if self.dataset.fps != self.fps:
                raise ValueError(
                    f"Existing dataset at {self.root} was recorded at {self.dataset.fps} fps, "
                    f"but dataset_fps is {self.fps}. Use a new dataset_repo_id/dataset_root."
                )
            self.features = self.dataset.meta.features
            n_cameras = sum(1 for key in self.features if key.startswith(RGB_PREFIX))
            if n_cameras:
                self.dataset.start_image_writer(
                    num_threads=self.image_writer_threads_per_camera * n_cameras
                )
            print(
                f"[recorder] Appending to {self.root} "
                f"({self.dataset.meta.total_episodes} episodes already saved)."
            )

    @property
    def num_saved_episodes(self) -> int:
        return 0 if self.dataset is None else self.dataset.meta.total_episodes

    def start_episode(self) -> None:
        if self.recording:
            raise RuntimeError("An episode is already being recorded.")
        self.recording = True
        self.episode_frames = 0

    def add_frame(
        self,
        datapoints: Sequence[Datapoint],
        snapshots: Sequence[RobotSnapshot] | RobotSnapshot,
        actions: Sequence[dict[str, float]] | dict[str, float],
    ) -> None:
        """Buffer one step of the current episode (images are written by background threads).

        ``snapshots``/``actions`` are per robot, or a single robot's snapshot and action."""
        if not self.recording:
            raise RuntimeError("Call start_episode() before add_frame().")
        snapshots, actions = _per_robot(snapshots, actions)
        features = build_features(datapoints, snapshots, actions)
        self._ensure_dataset(features, datapoints, snapshots)
        self.dataset.add_frame(build_frame(datapoints, snapshots, actions, self.features, self.task))
        self.episode_frames += 1

    def stop_episode(self) -> int:
        """Stop buffering frames; returns the episode's frame count. Then save or discard it."""
        self.recording = False
        return self.episode_frames

    def save_episode(self) -> None:
        self.recording = False
        if self.episode_frames == 0:
            return
        self.dataset.save_episode()
        self.episode_frames = 0

    def discard_episode(self) -> None:
        self.recording = False
        if self.dataset is not None and self.episode_frames > 0:
            self.dataset.clear_episode_buffer()
        self.episode_frames = 0

    def finalize(self) -> None:
        """Drop any unsaved episode, then close LeRobot's writers so the dataset is loadable."""
        if self.episode_frames:
            self.discard_episode()
        if self.dataset is None:
            return
        self.dataset.finalize()
        self.dataset.stop_image_writer()

    def _ensure_dataset(self, features, datapoints, snapshots) -> None:
        if self.dataset is not None:
            if _comparable(features) != _comparable(self.features):
                raise ValueError(
                    f"This step's data doesn't match the dataset at {self.root} (different "
                    "cameras, image size, robot links, or joints). Record into a new "
                    "dataset_repo_id/dataset_root instead."
                )
            return
        n_cameras = len(datapoints)
        self.dataset = LeRobotDataset.create(
            self.repo_id,
            self.fps,
            features,
            root=self.root,
            robot_type=self.robot_type,
            use_videos=True,
            image_writer_threads=self.image_writer_threads_per_camera * n_cameras,
        )
        self.features = self.dataset.meta.features
        self._write_sidecar(datapoints, snapshots)
        print(f"[recorder] Created dataset at {self.root}.")

    def _write_sidecar(self, datapoints, snapshots) -> None:
        world_frame = (
            "YAM station URDF world; robot link clouds and extrinsics (X_WC, camera -> world) are in it"
            if self.robot_type == "yam_follower"
            else "robot base (robot 0); extrinsics are X_WC (camera -> world)"
        )
        sidecar = {
            "format_version": SIDECAR_FORMAT_VERSION,
            "depth_encoding": DEPTH_ENCODING,
            "depth_units": "raw sensor units; meters = depth * observation.depth_scale.<serial>",
            "rgb_channel_order": "rgb",
            "depth_aligned_to": "color",
            "world_frame": world_frame,
            "camera_serials": [dp.serial for dp in datapoints],
            "robot_links": list(world_link_pcds(snapshots)),
            "joint_names": list(_merged([s.joint_positions for s in snapshots], "joint names")),
        }
        path = self.root / "meta" / SIDECAR_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sidecar, indent=2))
