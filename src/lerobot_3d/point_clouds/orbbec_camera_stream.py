"""orbbec_camera_stream.py

Orbbec Femto Bolt RGB-D camera stream for the lerobot_3d pipeline.
Analogous to camera_stream.py (RealSense version) but for the
Orbbec Femto Bolt using pyorbbecsdk2.

Key differences from RealSense version:
- MJPG color stream requires cv2.imdecode instead of direct array cast
- D2C hardware alignment via AlignFilter (vs rs.align)
- Frame buffer + capture thread handles depth(15fps)/color(30fps) mismatch
- Color intrinsics read from SDK at native resolution, scaled to stream res
- OBIntrinsicsCompat adapts Orbbec cx/cy naming to RealSense ppx/ppy
  so get_fused_point_cloud() from camera_stream.py works unchanged

Hardware: Orbbec Femto Bolt, USB 3.0+


*Note: - Depth values returned in metres (depth_scale=1.0, max_depth=5.0)
  to match the RealSense convention used by get_fused_point_cloud()
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

from lerobot_3d.common.types import Datapoint
from lerobot_3d.paths import resolve_extrinsic_calibration_json
from lerobot_3d.point_clouds.camera_stream import get_fused_point_cloud
from lerobot_3d.point_clouds.point_cloud_viewer import LivePointCloudViewer

# --- Stream configuration ---
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 576
DEPTH_FPS = 15

COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
COLOR_FPS = 30

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 5000
DEPTH_MIN_M = DEPTH_MIN_MM / 1000.0   # 0.1 m
DEPTH_MAX_M = DEPTH_MAX_MM / 1000.0   # 5.0 m
UPDATE_EVERY_N_FRAMES = 10

# Drop color frames that take longer than this to decode.
# Prevents shadow artifact from MJPG decode spikes on Windows.
MAX_DECODE_MS = 25


@dataclass
class OBIntrinsicsCompat:
    """Wraps Orbbec intrinsics to match the RealSense naming convention.

    RealSense uses .ppx / .ppy for the principal point.
    Orbbec uses .cx / .cy.
    get_fused_point_cloud() in camera_stream.py expects .ppx / .ppy,
    so this shim makes Orbbec intrinsics a drop-in replacement.
    """

    fx: float
    fy: float
    ppx: float  # cx
    ppy: float  # cy
    width: int
    height: int


class FrameBuffer:
    """Thread-safe buffer holding the most recent depth and color frames.

    Depth runs at 15 fps, color at 30 fps. Rather than waiting for a
    perfectly synchronised pair (which causes lag), the buffer always
    exposes the freshest available frame from each stream independently.
    This is standard practice for sensors running at different rates.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth_mm: np.ndarray | None = None
        self._color_bgr: np.ndarray | None = None
        self._depth_count = 0
        self._color_count = 0
        self._dropped_count = 0

    def update_depth(self, depth_mm: np.ndarray) -> None:
        with self._lock:
            self._depth_mm = depth_mm.copy()
            self._depth_count += 1

    def update_color(self, color_bgr: np.ndarray) -> None:
        with self._lock:
            self._color_bgr = color_bgr.copy()
            self._color_count += 1

    def increment_dropped(self) -> None:
        with self._lock:
            self._dropped_count += 1

    def get_latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (depth_mm, color_bgr) or (None, None) if not ready."""
        with self._lock:
            if self._depth_mm is None or self._color_bgr is None:
                return None, None
            return self._depth_mm.copy(), self._color_bgr.copy()

    def get_stats(self) -> tuple[int, int, int]:
        with self._lock:
            return self._depth_count, self._color_count, self._dropped_count


def _decode_color_frame(color_frame) -> np.ndarray | None:
    """Decode an MJPG color frame to a BGR numpy array."""
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _get_depth_colormap(depth_data: np.ndarray) -> np.ndarray:
    """Convert a depth array to a TURBO colormap image. Invalid pixels are black."""
    depth_clipped = np.clip(depth_data, DEPTH_MIN_MM, DEPTH_MAX_MM)
    depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    colormap[depth_data == 0] = 0
    return colormap


def _scale_intrinsics(
    intrinsics,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> OBIntrinsicsCompat:
    """Scale camera intrinsics proportionally when changing resolution.

    Focal lengths and principal point scale linearly with image dimensions.
    Returns an OBIntrinsicsCompat so the result is compatible with
    get_fused_point_cloud() from camera_stream.py.
    """
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return OBIntrinsicsCompat(
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        ppx=intrinsics.cx * scale_x,
        ppy=intrinsics.cy * scale_y,
        width=dst_w,
        height=dst_h,
    )


def _capture_loop(
    pipeline: Pipeline,
    align_filter: AlignFilter,
    frame_buffer: FrameBuffer,
    stop_event: threading.Event,
) -> None:
    """Background thread: capture frames continuously and update the buffer.

    Color frames whose MJPG decode time exceeds MAX_DECODE_MS are dropped
    to reduce the shadow artifact caused by decode spikes on Windows.
    """
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue

            try:
                aligned = align_filter.process(frames)
                if aligned is None:
                    aligned = frames
            except Exception:
                aligned = frames

            depth_frame = aligned.get_depth_frame()
            if depth_frame is not None:
                depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width())
                )
                depth_mm = depth_raw.astype(np.float32) * depth_frame.get_depth_scale()
                frame_buffer.update_depth(depth_mm)

            color_frame = aligned.get_color_frame()
            if color_frame is not None:
                t0 = time.perf_counter()
                color_bgr = _decode_color_frame(color_frame)
                decode_ms = 1000 * (time.perf_counter() - t0)

                if decode_ms > MAX_DECODE_MS:
                    frame_buffer.increment_dropped()
                elif color_bgr is not None:
                    frame_buffer.update_color(color_bgr)

        except Exception as e:  # noqa: BLE001
            if not stop_event.is_set():
                print(f"Capture error: {e}")
            break


class MultiOrbbecStream:
    """RGB-D stream for one or more Orbbec Femto Bolt cameras.

    Analogous to MultiRealSenseStream in camera_stream.py.
    Each camera runs a background capture thread that keeps a FrameBuffer
    up to date. get_datapoints() returns one Datapoint per camera, ready
    for get_fused_point_cloud() from camera_stream.py.

    Args:
        serial_numbers: List of Orbbec device serial numbers,
            e.g. ``["CL8346400LD"]``.  Pass an empty list to open
            the first available device.
        extrinsics_file: Path to a JSON file with per-serial X_WC matrices,
            in the same format used by MultiRealSenseStream.
        width: Color stream width in pixels.
        height: Color stream height in pixels.
        fps: Color stream frame rate.
    """

    def __init__(
        self,
        serial_numbers: list[str],
        extrinsics_file: str,
        width: int = COLOR_WIDTH,
        height: int = COLOR_HEIGHT,
        fps: int = COLOR_FPS,
    ) -> None:
        self.serial_numbers = serial_numbers
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.extrinsics_file = str(resolve_extrinsic_calibration_json(extrinsics_file))
        self.get_camera_extrinsics()

        self._pipelines: dict[str, Pipeline] = {}
        self._buffers: dict[str, FrameBuffer] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._intrinsics: dict[str, OBIntrinsicsCompat] = {}

        # Use a single serial placeholder when none are specified
        device_keys = serial_numbers if serial_numbers else ["default"]

        for serial in device_keys:
            pipeline = Pipeline()
            config = Config()

            # Depth stream
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profiles.get_video_stream_profile(
                DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS
            )
            config.enable_stream(depth_profile)

            # Color stream
            color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_video_stream_profile(
                self.width, self.height, OBFormat.MJPG, self.fps
            )
            config.enable_stream(color_profile)

            pipeline.start(config)

            # Scale color intrinsics from native sensor resolution
            # to the requested stream resolution
            camera_param = pipeline.get_camera_param()
            color_intr = camera_param.rgb_intrinsic
            self._intrinsics[serial] = _scale_intrinsics(
                color_intr,
                color_intr.width,
                color_intr.height,
                self.width,
                self.height,
            )

            # D2C alignment: reprojects depth into color camera frame
            align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

            # Per-camera frame buffer and capture thread
            buf = FrameBuffer()
            stop = threading.Event()
            thread = threading.Thread(
                target=_capture_loop,
                args=(pipeline, align_filter, buf, stop),
                daemon=True,
            )
            thread.start()

            self._pipelines[serial] = pipeline
            self._buffers[serial] = buf
            self._stop_events[serial] = stop
            self._threads[serial] = thread

    def get_camera_extrinsics(self) -> None:
        """Load per-camera extrinsic matrices from the JSON calibration file.

        Expects the same format as MultiRealSenseStream:
        ``{ "SERIAL": { "X_WC": [[...], ...] }, ... }``
        """
        with open(self.extrinsics_file) as f:
            e = json.load(f)

        self.extrinsics: dict[str, dict[str, np.ndarray]] = {}
        for serial, data in e.items():
            self.extrinsics[serial] = {"X_WC": np.array(data["X_WC"])}

    def get_datapoints(self) -> list[Datapoint]:
        """Return one Datapoint per camera with the latest RGB-D frames.

        Depth values are in millimetres (depth_scale = 1.0).
        color_intrinsics uses .ppx / .ppy naming for compatibility with
        get_fused_point_cloud() from camera_stream.py.
        """
        datapoints = []

        for serial, buf in self._buffers.items():
            depth_mm, color_bgr = buf.get_latest()

            if depth_mm is None or color_bgr is None:
                continue

            # Convert depth mm -> metres to match RealSense convention
            depth = depth_mm.copy().astype(np.float32) / 1000.0
            depth[depth < DEPTH_MIN_M] = 0.0
            depth[depth > DEPTH_MAX_M] = 0.0

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            x_wc = self.extrinsics.get(serial, {}).get("X_WC", np.eye(4))

            datapoints.append(
                Datapoint(
                    serial=serial,
                    color=color_rgb,
                    depth=depth,
                    depth_scale=1.0,        # depth already in metres
                    max_depth=DEPTH_MAX_M,  # 5.0 m
                    X_WC=x_wc,
                    color_intrinsics=self._intrinsics[serial],
                    obj_mask=None,
                )
            )

        return datapoints

    def stop(self) -> None:
        """Stop all capture threads and pipelines."""
        for stop in self._stop_events.values():
            stop.set()
        for thread in self._threads.values():
            thread.join(timeout=2.0)
        for pipeline in self._pipelines.values():
            pipeline.stop()


if __name__ == "__main__":
    # Standalone viewer — mirrors the __main__ block in camera_stream.py.
    # Replace the serial number with your device's SN (printed on the camera).
    # Leave the list empty to open the first available device.
    serials = ["CL8346400LD"]
    stream = MultiOrbbecStream(serials, "extrinsic_calibration.json")
    pcd_viewer = LivePointCloudViewer()

    for _ in range(1000):
        datapoints = stream.get_datapoints()
        if not datapoints:
            continue
        fused, _ = get_fused_point_cloud(datapoints)
        pts = np.asarray(fused.points)
        cols = np.asarray(fused.colors) if fused.has_colors() else None
        pcd_viewer.update(pts, cols)

    stream.stop()