"""The table's plane in the world (station) frame, fit to SAM-segmented depth from every camera.

Downstream planners (e.g. a BOBA world model whose ground is ``z = 0``) only need where the table is:
:attr:`TablePlane.X_W_table` is a frame on the table top with ``+z`` along its upward normal, so
``inv(X_W_table)`` maps world points to heights above the table. It's stored as the ``table_plane``
entry of the extrinsics file, next to the cameras and ``yam_<side>_base``; read it back with
:func:`table_plane_from_extrinsics` or :func:`load_table_plane`.

Fitting is RANSAC (robust to SAM mask spill onto objects and flying pixels at depth edges)
followed by a least-squares refit on the inliers. Pure numpy/OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot_3d.point_clouds.rgbd import depth_to_points, transform_points

TABLE_PLANE_KEY = "table_plane"
"""Extrinsics-file entry holding the fitted table (see :meth:`TablePlane.to_json`)."""
DISTANCE_THRESHOLD_M = 0.004
"""RANSAC inlier distance: a few mm, about RealSense/Orbbec depth noise at arm's length."""
EXTENT_CELL_M = 0.01
"""Grid cell for finding the table top's region among the inliers (see :func:`_table_region`)."""


@dataclass
class TablePlane:
    """The table top: plane ``normal . x + offset = 0`` (world frame, ``normal`` up) and its extent."""

    normal: np.ndarray
    """``(3,)`` unit normal pointing up, away from the table."""
    offset: float
    X_W_table: np.ndarray
    """``(4, 4)`` table frame: origin at the table top's center, ``+z`` = ``normal``, ``+x`` along its long edge."""
    extent: np.ndarray
    """``(2,)`` table top size (m) along the table frame's x and y (inliers' minimum-area rectangle)."""
    rms: float
    """RMS distance (m) of the inliers from the plane."""
    inliers: int

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Height (m) of each world point above the table plane (negative = below)."""
        return np.asarray(points, dtype=np.float64) @ self.normal + self.offset

    def corners(self) -> np.ndarray:
        """``(4, 3)`` world corners of the table top, counter-clockwise seen from above."""
        hx, hy = 0.5 * self.extent
        local = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]])
        return transform_points(self.X_W_table, local)

    def to_json(self) -> dict:
        return {
            "X_W_table": self.X_W_table.tolist(),
            "normal": self.normal.tolist(),
            "offset": float(self.offset),
            "extent": self.extent.tolist(),
            "rms": float(self.rms),
            "inliers": int(self.inliers),
        }

    @classmethod
    def from_json(cls, entry: dict) -> TablePlane:
        return cls(
            normal=np.asarray(entry["normal"], dtype=np.float64),
            offset=float(entry["offset"]),
            X_W_table=np.asarray(entry["X_W_table"], dtype=np.float64),
            extent=np.asarray(entry["extent"], dtype=np.float64),
            rms=float(entry["rms"]),
            inliers=int(entry["inliers"]),
        )


def table_plane_from_extrinsics(extrinsics: dict) -> TablePlane | None:
    """The fitted table in an extrinsics dict, or ``None`` if it hasn't been fit."""
    entry = extrinsics.get(TABLE_PLANE_KEY)
    return None if entry is None else TablePlane.from_json(entry)


def load_table_plane(extrinsic_json: str | Path) -> TablePlane | None:
    """The fitted table from the extrinsics file teleop reads (e.g. ``config.extrinsic_json``)."""
    from lerobot_3d.point_clouds.ros_camera_stream import extrinsics_path, read_extrinsics

    return table_plane_from_extrinsics(read_extrinsics(extrinsics_path(extrinsic_json)))


def masked_world_points(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    X_WC: np.ndarray,
    *,
    erode_px: int = 5,
    stride: int = 2,
    max_depth: float | None = None,
) -> np.ndarray:
    """World points of ``depth`` (registered to the image ``mask`` was drawn on, intrinsics ``K``)
    inside ``mask``, eroded by ``erode_px`` so mask edges spilling onto other surfaces are dropped."""
    mask = np.asarray(mask, dtype=np.uint8)
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    masked = np.where(mask > 0, depth, 0)
    points = depth_to_points(masked, K, depth_scale, stride=stride, max_depth=max_depth)
    return transform_points(X_WC, points)


def _plane_through(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane ``(unit normal, offset)`` through ``points`` (smallest singular vector)."""
    centroid = points.mean(axis=0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][-1]
    return normal, -float(normal @ centroid)


def _ransac(points: np.ndarray, threshold: float, iterations: int, rng: np.random.Generator,
            max_scored: int = 20000) -> tuple[np.ndarray, float]:
    """Plane with the most points within ``threshold``, scored on at most ``max_scored`` points."""
    scored = points if len(points) <= max_scored else points[rng.choice(len(points), max_scored, replace=False)]
    samples = points[rng.integers(0, len(points), (iterations, 3))]
    normals = np.cross(samples[:, 1] - samples[:, 0], samples[:, 2] - samples[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > 1e-9  # drop degenerate (collinear) samples
    normals = normals[keep] / norms[keep, None]
    offsets = -np.einsum("ij,ij->i", normals, samples[keep, 0])
    if not len(normals):
        raise ValueError("All RANSAC samples were degenerate; the points are (nearly) collinear")
    best, best_count = 0, -1
    for start in range(0, len(normals), 64):  # chunks keep the (chunk, scored) distance matrix small
        counts = (np.abs(scored @ normals[start:start + 64].T + offsets[start:start + 64]) < threshold).sum(axis=0)
        if counts.max() > best_count:
            best, best_count = start + int(counts.argmax()), int(counts.max())
    return normals[best], float(offsets[best])


def _table_region(uv: np.ndarray, cell: float) -> np.ndarray:
    """The ``(N, 2)`` in-plane points forming the largest connected patch, on a ``cell``-sized grid.

    Inliers also include stray points that just happen to lie near the plane (mask spill, flying
    pixels, other surfaces at table height); they'd stretch a bounding rectangle, so isolated cells
    are opened away and only the biggest component is kept.
    """
    lo = uv.min(axis=0)
    ij = np.floor((uv - lo) / cell).astype(np.int64)
    shape = ij.max(axis=0) + 1
    if shape.prod() > 4096 ** 2:
        raise ValueError(f"Plane inliers span {shape * cell} m; is the mask on the table?")
    grid = np.zeros((shape[1], shape[0]), dtype=np.uint8)
    grid[ij[:, 1], ij[:, 0]] = 1
    opened = cv2.morphologyEx(grid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened if opened.any() else grid, connectivity=8)
    if count < 2:
        return uv
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return uv[labels[ij[:, 1], ij[:, 0]] == largest]


def fit_table_plane(
    points: np.ndarray,
    *,
    up=(0.0, 0.0, 1.0),
    distance_threshold: float = DISTANCE_THRESHOLD_M,
    iterations: int = 1000,
    extent_cell: float = EXTENT_CELL_M,
    seed: int = 0,
) -> TablePlane:
    """Fit the table top to ``(N, 3)`` world points (mostly table; outliers are fine).

    ``up`` is roughly the world's up direction; the normal is flipped to agree with it.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError(f"Need at least 3 points to fit a plane, got {len(points)}")
    normal, offset = _ransac(points, distance_threshold, iterations, np.random.default_rng(seed))
    for _ in range(2):  # refit on inliers, then once more on the refit's inliers
        inliers = points[np.abs(points @ normal + offset) < distance_threshold]
        if len(inliers) < 3:
            raise ValueError("Too few plane inliers; is the mask on the table?")
        normal, offset = _plane_through(inliers)
    if normal @ np.asarray(up, dtype=np.float64) < 0:
        normal, offset = -normal, -offset
    distances = inliers @ normal + offset

    # In-plane basis to find the table top's minimum-area rectangle.
    seed_axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, seed_axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    on_plane = inliers - distances[:, None] * normal
    uv = _table_region(on_plane @ np.stack([u, v], axis=1), extent_cell)
    (cu, cv), (wu, wv), angle = cv2.minAreaRect(uv.astype(np.float32))
    theta = np.deg2rad(angle)
    x_axis = np.cos(theta) * u + np.sin(theta) * v
    extent = np.array([wu, wv], dtype=np.float64)
    if wv > wu:  # +x along the long edge
        x_axis, extent = np.cross(normal, x_axis), extent[::-1].copy()
    X_W_table = np.eye(4)
    X_W_table[:3, 0] = x_axis
    X_W_table[:3, 1] = np.cross(normal, x_axis)
    X_W_table[:3, 2] = normal
    X_W_table[:3, 3] = cu * u + cv * v - offset * normal  # rectangle center, on the plane
    return TablePlane(
        normal=normal,
        offset=offset,
        X_W_table=X_W_table,
        extent=extent,
        rms=float(np.sqrt(np.mean(distances ** 2))),
        inliers=len(inliers),
    )
