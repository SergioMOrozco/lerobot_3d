"""
pointcloud_orbbec_final.py

Orbbec Femto Bolt - RGB-D point cloud viewer
Combines correct intrinsics from old script with
threading from new script.

Key design decisions:
- D2C alignment: depth reprojected into color camera frame
- Color intrinsics scaled from NATIVE 1920x1080 to working res
  (NOT from profile.get_width() which returns native, not requested)
- Frame buffer + capture thread: decouples capture from display
- MJPG spike frames dropped to reduce shadow artifact
- Working resolution: 1280x720

Hardware: Orbbec Femto Bolt, Windows laptop, USB 3.2
Author: Gabriel Piris
Date: 7/17/2026

Controls:
    s - save point cloud as .ply file
    q - quit
"""

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat,
    AlignFilter, OBStreamType
)
import numpy as np
import cv2
import open3d as o3d
import threading
import time


# --- Configuration ---
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 576
DEPTH_FPS = 15

# Color stream requested resolution
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
COLOR_FPS = 30

# Native color sensor resolution — used for intrinsics scaling
# Do NOT change this — it's a hardware constant for the Femto Bolt
#COLOR_NATIVE_WIDTH = 1920
#COLOR_NATIVE_HEIGHT = 1080

# Point cloud working resolution — matches color stream
WORK_WIDTH = COLOR_WIDTH
WORK_HEIGHT = COLOR_HEIGHT

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 5000
UPDATE_EVERY_N_FRAMES = 5

# Drop color frames that take longer than this to decode
# Prevents shadow artifact from MJPG decode spikes
MAX_DECODE_MS = 25


class FrameBuffer:
    """
    Thread-safe buffer holding the most recent depth and
    color frames independently.

    Depth runs at 15fps, color at 30fps. Rather than waiting
    for a perfectly synchronized pair (which causes lag), we
    always use the freshest available frame from each stream.

    This is standard practice in robotics systems where sensors
    run at different rates.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._depth_mm = None
        self._color_bgr = None
        self._depth_count = 0
        self._color_count = 0
        self._dropped_count = 0

    def update_depth(self, depth_mm):
        with self._lock:
            self._depth_mm = depth_mm.copy()
            self._depth_count += 1

    def update_color(self, color_bgr):
        with self._lock:
            self._color_bgr = color_bgr.copy()
            self._color_count += 1

    def increment_dropped(self):
        with self._lock:
            self._dropped_count += 1

    def get_latest(self):
        """Returns (depth_mm, color_bgr) or (None, None)."""
        with self._lock:
            if self._depth_mm is None or self._color_bgr is None:
                return None, None
            return self._depth_mm.copy(), self._color_bgr.copy()

    def get_stats(self):
        with self._lock:
            return (
                self._depth_count,
                self._color_count,
                self._dropped_count
            )


def scale_intrinsics(intrinsics, src_w, src_h, dst_w, dst_h):
    """
    Scale camera intrinsics proportionally when changing resolution.

    Focal lengths and principal point scale linearly with
    image dimensions. This must be called with the TRUE native
    resolution as src, not the profile's reported resolution.

    Args:
        intrinsics: SDK intrinsics object (fx, fy, cx, cy)
        src_w, src_h: TRUE native sensor resolution
        dst_w, dst_h: target working resolution

    Returns:
        dict with scaled fx, fy, cx, cy, width, height
    """
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return {
        "fx": intrinsics.fx * scale_x,
        "fy": intrinsics.fy * scale_y,
        "cx": intrinsics.cx * scale_x,
        "cy": intrinsics.cy * scale_y,
        "width": dst_w,
        "height": dst_h
    }


def decode_color_frame(color_frame):
    """Decode MJPG color frame to BGR. Returns None on failure."""
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return bgr


def get_depth_colormap(depth_data):
    """
    Convert depth array to TURBO colormap visualization.
    Invalid pixels shown as black.
    """
    depth_clipped = np.clip(depth_data, DEPTH_MIN_MM, DEPTH_MAX_MM)
    depth_norm = cv2.normalize(
        depth_clipped, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    colormap[depth_data == 0] = 0
    return colormap


def get_point_cloud(depth_mm, color_rgb, intr):
    """
    Generate colored point cloud via Open3D RGBD pipeline.
    Matches lerobot_3d architecture for future compatibility.

    After D2C alignment, depth is in color camera frame.
    Must use color intrinsics (scaled to working resolution).

    Args:
        depth_mm: HxW float32 depth in millimeters
        color_rgb: HxW x3 uint8 RGB (same size as depth)
        intr: dict with fx, fy, cx, cy, width, height

    Returns:
        Open3D PointCloud in camera frame (X_WC = identity)
    """
    h, w = depth_mm.shape
    assert w == intr["width"] and h == intr["height"], \
        f"Size mismatch: image={w}x{h} intrinsics={intr['width']}x{intr['height']}"

    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        w, h,
        intr["fx"], intr["fy"],
        intr["cx"], intr["cy"]
    )

    # Open3D works in meters
    depth_m = (depth_mm / 1000.0).astype(np.float32)

    depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_m))
    color_img = o3d.geometry.Image(np.ascontiguousarray(color_rgb))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img,
        depth_img,
        depth_scale=1.0,
        depth_trunc=DEPTH_MAX_MM / 1000.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, o3d_intr
    )

    # X_WC = identity for now
    # Replace with calibrated extrinsic matrix when available
    pcd.transform(np.eye(4))

    return pcd


def capture_loop(pipeline, align_filter, frame_buffer, stop_event):
    """
    Runs in background thread.
    Captures frames continuously and updates the buffer.
    Drops color frames with MJPG decode spikes > MAX_DECODE_MS
    to reduce shadow artifact.
    """
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue

            # Apply D2C alignment
            try:
                aligned = align_filter.process(frames)
                if aligned is None:
                    aligned = frames
            except Exception:
                aligned = frames

            # Update depth buffer
            depth_frame = aligned.get_depth_frame()
            if depth_frame is not None:
                depth_raw = np.frombuffer(
                    depth_frame.get_data(), dtype=np.uint16
                ).reshape((
                    depth_frame.get_height(),
                    depth_frame.get_width()
                ))
                scale = depth_frame.get_depth_scale()
                depth_mm = depth_raw.astype(np.float32) * scale
                frame_buffer.update_depth(depth_mm)

            # Update color buffer — drop frames with decode spikes
            color_frame = aligned.get_color_frame()
            if color_frame is not None:
                t0 = time.perf_counter()
                color_bgr = decode_color_frame(color_frame)
                decode_ms = 1000 * (time.perf_counter() - t0)

                if decode_ms > MAX_DECODE_MS:
                    # Spike frame — keep previous color, don't update
                    frame_buffer.increment_dropped()
                elif color_bgr is not None:
                    frame_buffer.update_color(color_bgr)

        except Exception as e:
            if not stop_event.is_set():
                print(f"Capture error: {e}")
            break


def main():
    pipeline = Pipeline()
    config = Config()

    # --- Depth stream ---
    try:
        depth_profiles = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        depth_profile = depth_profiles.get_video_stream_profile(
            DEPTH_WIDTH, DEPTH_HEIGHT, OBFormat.Y16, DEPTH_FPS
        )
        config.enable_stream(depth_profile)
        print(f"Depth:  {DEPTH_WIDTH}x{DEPTH_HEIGHT} @ {DEPTH_FPS}fps Y16")
    except Exception as e:
        print(f"Depth setup error: {e}")
        return

    # --- Color stream ---
    # Request COLOR_WIDTH x COLOR_HEIGHT explicitly
    # Do NOT use get_width()/get_height() after this —
    # they return native resolution (1920x1080), not requested res
    try:
        color_profiles = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        color_profile = color_profiles.get_video_stream_profile(
            COLOR_WIDTH, COLOR_HEIGHT, OBFormat.MJPG, COLOR_FPS
        )
        config.enable_stream(color_profile)
        print(f"Color:  {COLOR_WIDTH}x{COLOR_HEIGHT} @ {COLOR_FPS}fps MJPG")
    except Exception as e:
        print(f"Color setup error: {e}")
        return

    pipeline.start(config)

    # --- Intrinsics ---
    camera_param = pipeline.get_camera_param()
    depth_intr = camera_param.depth_intrinsic
    color_intr = camera_param.rgb_intrinsic

    print(f"\nDepth intrinsics (native {depth_intr.width}x{depth_intr.height}):")
    print(f"  fx={depth_intr.fx:.2f}  fy={depth_intr.fy:.2f}")
    print(f"  cx={depth_intr.cx:.2f}  cy={depth_intr.cy:.2f}")

    print(f"\nColor intrinsics (native {color_intr.width}x{color_intr.height}):")
    print(f"  fx={color_intr.fx:.2f}  fy={color_intr.fy:.2f}")
    print(f"  cx={color_intr.cx:.2f}  cy={color_intr.cy:.2f}")

    # Scale color intrinsics from NATIVE resolution to working res
    # CRITICAL: use COLOR_NATIVE_WIDTH/HEIGHT (1920x1080), not
    # profile.get_width() which returns native regardless of request
    color_native_w = color_intr.width
    color_native_h = color_intr.height
    print(f"Color native resolution from SDK: {color_native_w}x{color_native_h}")

    work_intr = scale_intrinsics(
        color_intr,
        color_native_w, color_native_h,
        WORK_WIDTH, WORK_HEIGHT
    )

    print(f"\nScaled color intrinsics ({WORK_WIDTH}x{WORK_HEIGHT}):")
    print(f"  fx={work_intr['fx']:.2f}  fy={work_intr['fy']:.2f}")
    print(f"  cx={work_intr['cx']:.2f}  cy={work_intr['cy']:.2f}")

    # Sanity check — cx and cy should be near image center
    cx_expected = WORK_WIDTH / 2
    cy_expected = WORK_HEIGHT / 2
    cx_ok = abs(work_intr["cx"] - cx_expected) < cx_expected * 0.2
    cy_ok = abs(work_intr["cy"] - cy_expected) < cy_expected * 0.2
    if not cx_ok or not cy_ok:
        print(f"\nWARNING: Principal point looks off.")
        print(f"  Expected cx~{cx_expected:.0f}, got {work_intr['cx']:.1f}")
        print(f"  Expected cy~{cy_expected:.0f}, got {work_intr['cy']:.1f}")
        print(f"  Check COLOR_NATIVE_WIDTH/HEIGHT constants.")
    else:
        print(f"  Intrinsics sanity check: OK")

    # --- D2C Alignment filter ---
    print("\nInitializing D2C alignment filter...")
    try:
        align_filter = AlignFilter(
            align_to_stream=OBStreamType.COLOR_STREAM
        )
        print("D2C alignment: ENABLED")
    except Exception as e:
        print(f"AlignFilter unavailable: {e}")
        print("Cannot continue without alignment.")
        return

    # --- Frame buffer + capture thread ---
    frame_buffer = FrameBuffer()
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(pipeline, align_filter, frame_buffer, stop_event),
        daemon=True
    )
    capture_thread.start()
    print("Capture thread started.")
    print(f"MJPG spike threshold: {MAX_DECODE_MS}ms")
    print(f"\nControls: s=save  q=quit\n")

    # --- Open3D viewer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        "Femto Bolt - Point Cloud", width=900, height=700
    )
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    pcd_geo = o3d.geometry.PointCloud()
    geometry_added = False
    display_frame = 0
    save_count = 0

    try:
        while True:
            depth_mm, color_bgr = frame_buffer.get_latest()

            if depth_mm is None or color_bgr is None:
                vis.poll_events()
                vis.update_renderer()
                cv2.waitKey(1)
                continue

            display_frame += 1

            # --- Display windows ---
            color_display = cv2.resize(color_bgr, (640, 360))
            cv2.imshow("Color Reference", color_display)

            depth_colormap = get_depth_colormap(depth_mm)
            depth_display = cv2.resize(depth_colormap, (640, 360))
            cv2.imshow("Depth Stream", depth_display)

            color_for_overlay = cv2.resize(
                color_bgr,
                (depth_mm.shape[1], depth_mm.shape[0])
            )
            overlay = cv2.addWeighted(
                color_for_overlay, 0.6,
                depth_colormap, 0.4, 0
            )
            overlay_display = cv2.resize(overlay, (640, 360))
            cv2.imshow("Alignment Overlay", overlay_display)

            # --- Point cloud update ---
            if display_frame % UPDATE_EVERY_N_FRAMES == 0:

                depth_work = cv2.resize(
                    depth_mm,
                    (WORK_WIDTH, WORK_HEIGHT),
                    interpolation=cv2.INTER_NEAREST
                )
                color_work = cv2.resize(
                    color_bgr, (WORK_WIDTH, WORK_HEIGHT)
                )
                color_rgb_work = cv2.cvtColor(
                    color_work, cv2.COLOR_BGR2RGB
                )

                depth_work[depth_work < DEPTH_MIN_MM] = 0
                depth_work[depth_work > DEPTH_MAX_MM] = 0

                new_pcd = get_point_cloud(
                    depth_work, color_rgb_work, work_intr
                )

                pts = np.asarray(new_pcd.points)
                cols = np.asarray(new_pcd.colors)

                if len(pts) > 0:
                    pts_mm = pts * 1000.0
                    d_count, c_count, dropped = frame_buffer.get_stats()
                    print(
                        f"Frame {display_frame}: "
                        f"{len(pts):,} pts | "
                        f"Z: {pts_mm[:,2].min():.0f}-"
                        f"{pts_mm[:,2].max():.0f}mm | "
                        f"d={d_count} c={c_count} "
                        f"dropped={dropped}"
                    )

                    pcd_geo.points = o3d.utility.Vector3dVector(pts)
                    pcd_geo.colors = o3d.utility.Vector3dVector(cols)

                    if not geometry_added:
                        vis.add_geometry(pcd_geo)
                        ctr = vis.get_view_control()
                        ctr.set_zoom(0.5)
                        ctr.set_front([0, 0, -1])
                        ctr.set_up([0, -1, 0])
                        ctr.set_lookat([0, 0, 2.0])
                        geometry_added = True
                    else:
                        vis.update_geometry(pcd_geo)

            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_count += 1
                path = f"pointcloud_{save_count:04d}.ply"
                o3d.io.write_point_cloud(path, pcd_geo)
                print(f"Saved: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        d, c, dropped = frame_buffer.get_stats()
        print(f"\nDone. depth={d} color={c} "
              f"dropped={dropped} saved={save_count}")


if __name__ == "__main__":
    main()