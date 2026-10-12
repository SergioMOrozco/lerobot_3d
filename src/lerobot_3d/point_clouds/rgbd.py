"""Depth/color helpers for streams that arrive as compressed images (e.g. over ROS).

Pure numpy/OpenCV: no camera SDK, no ROS. Frames follow the optical convention
(x right, y down, z forward).
"""
from __future__ import annotations

import numpy as np

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def decode_color(fmt: str, data: bytes) -> np.ndarray:
    """JPEG/PNG ``sensor_msgs/CompressedImage`` payload -> ``(H, W, 3)`` uint8 **RGB**."""
    import cv2

    image = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode color image (format {fmt!r})")
    return np.ascontiguousarray(image[:, :, ::-1])


def decode_depth(fmt: str, data: bytes) -> np.ndarray:
    """Lossless ``compressedDepth`` (PNG behind a 12-byte header) or plain PNG -> ``(H, W)`` uint16."""
    import cv2

    payload = bytes(data)
    if payload[12:20] == _PNG_MAGIC:
        payload = payload[12:]
    elif payload[:8] != _PNG_MAGIC:
        raise ValueError(f"Expected lossless PNG depth, got format {fmt!r} (RVL/JPEG depth not supported)")
    depth = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Depth PNG did not decode to a single-channel uint16 image")
    return depth


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    *,
    stride: int = 1,
    max_depth: float | None = None,
) -> np.ndarray:
    """Back-project a depth image into ``(N, 3)`` metres in the depth camera's optical frame.

    Zero (invalid) pixels and pixels beyond ``max_depth`` are dropped. ``stride`` subsamples
    rows/columns. ``K`` is the 3x3 pinhole matrix of the *depth* image.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    stride = max(1, int(stride))
    z = depth[::stride, ::stride].astype(np.float32) * float(depth_scale)
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    valid = z > 0
    if max_depth is not None:
        valid &= z <= max_depth
    z, u, v = z[valid], u[valid].astype(np.float32), v[valid].astype(np.float32)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def sample_colors(points_in_color: np.ndarray, color_rgb: np.ndarray, K_color: np.ndarray) -> np.ndarray:
    """Color for each point (already in the color camera's optical frame) by projecting into
    ``color_rgb``. Points behind the camera or outside the image get mid-gray."""
    K = np.asarray(K_color, dtype=np.float64).reshape(3, 3)
    colors = np.full((len(points_in_color), 3), 128, dtype=np.uint8)
    if len(points_in_color) == 0:
        return colors
    z = points_in_color[:, 2]
    front = z > 1e-6
    safe_z = np.where(front, z, 1.0)
    u = np.round(points_in_color[:, 0] * K[0, 0] / safe_z + K[0, 2]).astype(np.int64)
    v = np.round(points_in_color[:, 1] * K[1, 1] / safe_z + K[1, 2]).astype(np.int64)
    h, w = color_rgb.shape[:2]
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    colors[inside] = color_rgb[v[inside], u[inside]]
    return colors


def colored_cloud(
    depth: np.ndarray,
    K_depth: np.ndarray,
    depth_scale: float,
    color_rgb: np.ndarray | None,
    K_color: np.ndarray | None,
    T_color_depth: np.ndarray,
    *,
    stride: int = 4,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth image -> ``(points, uint8 colors)`` expressed in the **color** optical frame.

    ``T_color_depth`` maps depth-frame points into the color frame (factory extrinsics), so the
    cloud can be placed with the color camera's pose. Without a color image points are gray.
    """
    points = transform_points(T_color_depth, depth_to_points(depth, K_depth, depth_scale,
                                                               stride=stride, max_depth=max_depth))
    if color_rgb is None or K_color is None:
        return points, np.full((len(points), 3), 128, dtype=np.uint8)
    return points, sample_colors(points, color_rgb, K_color)
