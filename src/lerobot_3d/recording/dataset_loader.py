"""Read frames recorded by :class:`~lerobot_3d.recording.dataset_recorder.LeRobotDatasetRecorder`
back into the live pipeline's types.

Example::

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_3d.point_clouds.camera_stream import get_fused_point_cloud
    from lerobot_3d.recording.dataset_loader import (
        frame_link_pcds,
        frame_to_datapoints,
        read_fixed_extrinsics,
    )

    root = "path/to/dataset"
    ds = LeRobotDataset("local/my_teleop", root=root)
    extrinsics = read_fixed_extrinsics(root)          # {serial: X_WC}, once per dataset
    item = ds[0]
    datapoints = frame_to_datapoints(item, extrinsics=extrinsics)  # same Datapoints as live
    scene_pcd, _ = get_fused_point_cloud(datapoints)  # world-frame fused cloud
    links = frame_link_pcds(item)                     # {link_name: (N, 3)} world frame
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from lerobot_3d.common.types import Datapoint
from lerobot_3d.recording.dataset_recorder import (
    DEPTH_PREFIX,
    DEPTH_SCALE_PREFIX,
    EXTRINSICS_PREFIX,
    INTRINSICS_PREFIX,
    LINK_PCD_PREFIX,
    RGB_PREFIX,
    SIDECAR_NAME,
    STATE_KEY,
    read_fixed_extrinsics,  # noqa: F401  (re-exported: read alongside the loaders here)
)
from lerobot_3d.recording.depth_codec import unpack_depth

MAX_DEPTH_M = 10.0
"""Depth truncation used by the live stream (``MultiRealSenseStream``)."""


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_to_bgr_uint8(image) -> np.ndarray:
    """LeRobot's loaded ``(3, H, W)`` float RGB in ``[0, 1]`` (or ``(H, W, 3)`` uint8 RGB)
    -> ``(H, W, 3)`` uint8 BGR, the live ``Datapoint.color`` layout."""
    image = _to_numpy(image)
    if image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[..., ::-1])


def frame_serials(item: Mapping) -> list[str]:
    """Camera serials present in a frame, in recording order."""
    return [key[len(RGB_PREFIX):] for key in item if key.startswith(RGB_PREFIX)]


def _frame_X_WC(item: Mapping, serial: str, extrinsics: Mapping[str, np.ndarray] | None) -> np.ndarray:
    """A camera's pose: its per-frame feature if it has one, else the dataset's fixed ``X_WC``."""
    key = f"{EXTRINSICS_PREFIX}{serial}"
    if key in item:
        return _to_numpy(item[key]).astype(np.float64)
    if extrinsics is None or serial not in extrinsics:
        raise KeyError(
            f"Camera {serial} has fixed extrinsics: pass frame_to_datapoints(..., "
            "extrinsics=read_fixed_extrinsics(root))."
        )
    return np.asarray(extrinsics[serial], dtype=np.float64)


def frame_to_datapoints(
    item: Mapping,
    serials: Sequence[str] | None = None,
    joint_names: Sequence[str] | None = None,
    extrinsics: Mapping[str, np.ndarray] | None = None,
) -> list[Datapoint]:
    """Rebuild one frame's per-camera :class:`Datapoint` objects from a ``LeRobotDataset`` item.

    ``color`` comes back as BGR uint8 and ``depth`` as raw uint16 -- exactly what the live
    stream produced -- so ``get_fused_point_cloud`` and mask-based segmentation work unchanged.
    Pass ``joint_names`` (``ds.meta.features["observation.state"]["names"]``) to also fill each
    datapoint's ``joint_positions``, as the live pipeline does. Fixed cameras' poses aren't in
    the frame: pass ``extrinsics`` (:func:`read_fixed_extrinsics`) to fill their ``X_WC``.
    """
    serials = list(serials) if serials is not None else frame_serials(item)
    joint_positions = (
        frame_joint_positions(item, joint_names) if joint_names is not None else None
    )
    datapoints = []
    for s in serials:
        K = _to_numpy(item[f"{INTRINSICS_PREFIX}{s}"]).astype(np.float64)
        datapoints.append(
            Datapoint(
                serial=s,
                color=_rgb_to_bgr_uint8(item[f"{RGB_PREFIX}{s}"]),
                depth=unpack_depth(item[f"{DEPTH_PREFIX}{s}"]),
                depth_scale=float(_to_numpy(item[f"{DEPTH_SCALE_PREFIX}{s}"]).reshape(-1)[0]),
                max_depth=MAX_DEPTH_M,
                X_WC=_frame_X_WC(item, s, extrinsics),
                color_intrinsics=SimpleNamespace(
                    fx=float(K[0, 0]), fy=float(K[1, 1]), ppx=float(K[0, 2]), ppy=float(K[1, 2])
                ),
                joint_positions=joint_positions,
                moving=f"{EXTRINSICS_PREFIX}{s}" in item,
            )
        )
    return datapoints


def frame_link_pcds(item: Mapping) -> dict[str, np.ndarray]:
    """``{link_name: (N, 3) float32}`` world-frame robot link point clouds for one frame."""
    return {
        key[len(LINK_PCD_PREFIX):]: _to_numpy(value).astype(np.float32)
        for key, value in item.items()
        if key.startswith(LINK_PCD_PREFIX)
    }


def frame_joint_positions(item: Mapping, joint_names: Sequence[str]) -> dict[str, float]:
    """Motor-space ``{"<joint>.pos": value}`` from ``observation.state`` (names from the
    dataset's ``meta.features["observation.state"]["names"]`` or the sidecar)."""
    state = _to_numpy(item[STATE_KEY]).reshape(-1)
    return {name: float(v) for name, v in zip(joint_names, state)}


def read_sidecar(root: str | Path) -> dict:
    """The recorder's ``meta/lerobot_3d.json`` (depth encoding, serials, fixed extrinsics,
    link/joint names)."""
    return json.loads((Path(root).expanduser() / "meta" / SIDECAR_NAME).read_text())
