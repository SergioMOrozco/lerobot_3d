#!/usr/bin/env python3
"""
Record color + depth from two Intel RealSense cameras (by serial number) and write:

data/
  color/
    0/  (camera 0 frames as PNG)
    1/  (camera 1 frames as PNG)
    0.mp4
    1.mp4
  depth/
    0/  (camera 0 depth frames as 16-bit PNG)
    1/  (camera 1 depth frames as 16-bit PNG)
  calibrate.pkl     # list of 4x4 extrinsics (X_WC) in camera index order
  metadata.json     # intrinsics list, serial numbers, fps, WH, frame_num, start_step, end_step

Usage:
  python record_realsense_two_cam.py \
    --extrinsics extrinsic_calibration.json \
    --intrinsics instrinsic_calibration.json \
    --out_dir data \
    --serials 044322073544 244622072067 \
    --num_frames 116 \
    --fps 30

Deps:
  pip install pyrealsense2 opencv-python numpy
"""

import os
import json
import time
import argparse
import pickle
from pathlib import Path

import numpy as np
import cv2

import imageio.v3 as iio  # or just import imageio as iio if you prefer older API
import imageio

try:
    import pyrealsense2 as rs
except ImportError as e:
    raise SystemExit(
        "pyrealsense2 not found. Install it (e.g. `pip install pyrealsense2`) "
        "or via Intel RealSense SDK packaging for your OS."
    ) from e


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def K_from_intrinsics_dict(d: dict) -> list:
    """Return 3x3 camera intrinsics matrix as nested lists."""
    fx = float(d["fl_x"])
    fy = float(d["fl_y"])
    cx = float(d["cx"])
    cy = float(d["cy"])
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extrinsics", type=str, required=True, help="extrinsic_calibration.json path")
    parser.add_argument("--intrinsics", type=str, required=True, help="instrinsic_calibration.json path")
    parser.add_argument("--out_dir", type=str, default="data", help="output folder (will be created)")
    parser.add_argument(
        "--serials",
        type=str,
        nargs=2,
        required=True,
        help="two serial numbers in desired camera index order: cam0 cam1",
    )
    parser.add_argument("--num_frames", type=int, default=116)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--align_depth_to_color",
        action="store_true",
        help="align depth to color for each camera before saving",
    )
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=30,
        help="throw away initial frames to let exposure settle",
    )
    parser.add_argument(
        "--timeout_ms",
        type=int,
        default=5000,
        help="frameset wait timeout per camera",
    )
    breakpoint()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    color_dir = out_dir / "color"
    depth_dir = out_dir / "depth"

    # Create folder structure
    ensure_dir(color_dir / "0")
    ensure_dir(color_dir / "1")
    ensure_dir(depth_dir / "0")
    ensure_dir(depth_dir / "1")

    # Load calibrations
    with open(args.extrinsics, "r") as f:
        extr = json.load(f)
    with open(args.intrinsics, "r") as f:
        intr = json.load(f)

    serials = list(args.serials)
    for s in serials:
        if s not in extr:
            raise SystemExit(f"Serial {s} not found in extrinsics JSON: {args.extrinsics}")
        if s not in intr:
            raise SystemExit(f"Serial {s} not found in intrinsics JSON: {args.intrinsics}")

    # Build calibrate.pkl payload: list of 4x4 X_WC in camera index order
    calibrate_list = []
    for s in serials:
        X_WC = np.array(extr[s]["X_WC"], dtype=np.float64)
        if X_WC.shape != (4, 4):
            raise SystemExit(f"Extrinsics for {s} has wrong shape {X_WC.shape}, expected (4,4)")
        calibrate_list.append(X_WC)

    # Build metadata intrinsics list in camera index order
    intrinsics_list = [K_from_intrinsics_dict(intr[s]) for s in serials]

    # Prepare RealSense pipelines
    pipelines = []
    aligns = []
    try:
        ctx = rs.context()
        connected = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
        missing = [s for s in serials if s not in connected]
        if missing:
            raise SystemExit(f"These serials are not currently connected/visible: {missing}. Found: {connected}")

        for cam_idx, serial in enumerate(serials):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

            profile = pipeline.start(config)

            # Optional: set a high accuracy preset for depth if supported
            try:
                dev = profile.get_device()
                depth_sensor = dev.first_depth_sensor()
                if depth_sensor.supports(rs.option.visual_preset):
                    # 3 is often "High Accuracy" on D400 series, but can vary; if it errors, ignore.
                    depth_sensor.set_option(rs.option.visual_preset, 3)
            except Exception:
                pass

            pipelines.append(pipeline)
            aligns.append(rs.align(rs.stream.color))

        # Warm up (exposure / auto settings)
        for _ in range(max(0, args.warmup_frames)):
            for cam_idx, pipeline in enumerate(pipelines):
                fs = pipeline.wait_for_frames(args.timeout_ms)
                if args.align_depth_to_color:
                    fs = aligns[cam_idx].process(fs)
                # discard

        ## Video writers
        #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #writers = [
        #    cv2.VideoWriter(str(color_dir / "0.mp4"), fourcc, args.fps, (args.width, args.height)),
        #    cv2.VideoWriter(str(color_dir / "1.mp4"), fourcc, args.fps, (args.width, args.height)),
        #]
        #if not writers[0].isOpened() or not writers[1].isOpened():
        #    raise SystemExit("Failed to open one of the mp4 VideoWriters (check codec support).")

        # ImageIO writers (libx264). Use yuv420p for maximum compatibility.
        mp4_writers = [
            imageio.get_writer(
                str(color_dir / "0.mp4"),
                fps=args.fps,
                codec="libx264",
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            ),
            imageio.get_writer(
                str(color_dir / "1.mp4"),
                fps=args.fps,
                codec="libx264",
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            ),
        ]

        frame_num = args.num_frames
        start_step = 0
        end_step = frame_num - 1

        # Capture loop
        t0 = time.time()
        for i in range(frame_num):
            for cam_idx, pipeline in enumerate(pipelines):
                frameset = pipeline.wait_for_frames(args.timeout_ms)
                if args.align_depth_to_color:
                    frameset = aligns[cam_idx].process(frameset)

                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError(f"Missing color/depth frame at i={i} for cam_idx={cam_idx}")

                color = np.asanyarray(color_frame.get_data())  # BGR uint8, shape (H,W,3)
                depth = np.asanyarray(depth_frame.get_data())  # uint16, shape (H,W)

                # Save PNG frames
                cv2.imwrite(str(color_dir / str(cam_idx) / f"{i}.png"), color)
                np.save(depth_dir / str(cam_idx) / f"{i}.npy", depth)

                ## Write to MP4
                #writers[cam_idx].write(color)

                # color is BGR from RealSense; convert to RGB for imageio
                rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                mp4_writers[cam_idx].append_data(rgb)

            if (i + 1) % 10 == 0:
                dt = time.time() - t0
                print(f"[{i+1}/{frame_num}] captured ({(i+1)/dt:.2f} fps aggregate loop)")

        ## Close writers
        #for w in writers:
        #    w.release()

        for w in mp4_writers:
            w.close()

        # Write calibrate.pkl
        with open(out_dir / "calibrate.pkl", "wb") as f:
            pickle.dump(calibrate_list, f)

        # Write metadata.json
        metadata = {
            "intrinsics": intrinsics_list,      # list of 3x3
            "serial_numbers": serials,          # camera index order
            "fps": int(args.fps),
            "WH": [int(args.width), int(args.height)],
            "frame_num": int(frame_num),
            "start_step": int(start_step),
            "end_step": int(end_step),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"Done. Wrote dataset to: {out_dir.resolve()}")

    finally:
        # Stop pipelines cleanly
        for p in pipelines:
            try:
                p.stop()
            except Exception:
                pass


if __name__ == "__main__":
    # example: python phystwin_capture.py --extrinsics extrinsic_calibration.json --intrinsics intrinsic_calibration.json --out_dir rope_push_sergio --serials 044322073544 244622072067 --num_frames 116 --fps 30 --align_depth_to_color 
    main()
