# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot_playground.hardware_config import TeleopSystemConfig
from lerobot_playground.point_clouds.system_vis import SystemStateViewer


class TeleopPointCloudSystem:

    def __init__(self, config: TeleopSystemConfig):
        self.config = config
        self.leaders = [
            SO101Leader(SO101LeaderConfig(port=ax.port, id=ax.id)) for ax in config.leaders
        ]
        self.viewer = SystemStateViewer(config)

    def connect(self) -> None:
        print("Connecting devices...")
        for leader in self.leaders:
            leader.connect()
        print("Connected.")

    def step(self, masks_by_serial=None) -> tuple[list[dict], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """One control cycle: read leaders, update followers and viewer, return sensor data.

        Args:
            masks_by_serial: Optional mapping ``{camera_serial: mask}`` or sequence of
                masks aligned with ``config.realsense_serials``. Nonzero/True mask
                pixels are kept in the fused scene point cloud.

        Returns:
            datapoints: Raw per-camera datapoints used to build the fused point cloud.
            scene_pcd: ``(N, 3)`` ``float64`` fused scene points in world frame.
            robot_pcd: ``(M, 3)`` ``float64`` sampled follower mesh in world frame (first follower).
            robot_link_pcds: per-link robot clouds keyed by URDF link name.

        Poll ``self.viewer.quit`` to know when to stop the outer loop, then call :meth:`close`.
        """
        actions = [leader.get_action() for leader in self.leaders]
        return self.viewer.update(*actions, masks_by_serial=masks_by_serial)

    def close(self) -> None:
        """Release cameras, robots, and recording resources."""
        self.viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop + point cloud viewer.")
    parser.add_argument(
        "--recording_name", type=str, default="", help="which config to load"
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=60.0,
        help="Main loop rate in Hz (default 60). Use 0 or negative for no sleep (full speed).",
    )
    parser.add_argument(
        "--extrinsic-json",
        type=str,
        default="extrinsic_calibration.json",
        help="Camera extrinsics JSON (cwd, LEROBOT_PLAYGROUND_EXTRINSIC_JSON, or src/ next to package).",
    )
    parser.add_argument(
        "--realsense-serial",
        dest="realsense_serials",
        action="append",
        default=None,
        metavar="SERIAL",
        help="RealSense device serial (repeat flag once per camera, order matches extrinsics JSON). "
        "Omit to use defaults from TeleopSystemConfig.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable the Tk capture/save control panel.",
    )
    parser.add_argument(
        "--visualization",
        choices=("foxglove", "open3d", "both", "none"),
        default="foxglove",
        help="Visualization backend: Foxglove publishing, Open3D point_cloud_viewer, both, or none.",
    )
    parser.add_argument("--camera-width", type=int, default=848, help="RealSense color/depth width.")
    parser.add_argument("--camera-height", type=int, default=480, help="RealSense color/depth height.")
    parser.add_argument("--camera-fps", type=int, default=60, help="RealSense color/depth FPS.")
    parser.add_argument(
        "--action-interpolation-duration",
        type=float,
        default=0.12,
        help="Seconds to blend to each new follower target. Use 0 to disable smoothing.",
    )
    parser.add_argument(
        "--action-command-hz",
        type=float,
        default=50.0,
        help="Follower command loop rate while smoothing is enabled.",
    )

    args = parser.parse_args()

    config = replace(
        TeleopSystemConfig(),
        extrinsic_json=args.extrinsic_json,
        recording_name=args.recording_name,
        tune=not args.no_tune,
        publish_to_foxglove=args.visualization in ("foxglove", "both"),
        display_point_cloud_viewer=args.visualization in ("open3d", "both"),
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        action_interpolation_duration_s=args.action_interpolation_duration,
        action_command_hz=args.action_command_hz,
    )
    if args.realsense_serials is not None:
        config = replace(config, realsense_serials=tuple(args.realsense_serials))

    system = TeleopPointCloudSystem(config)

    system.connect()

    period_s = None if args.hz is None or args.hz <= 0 else 1.0 / args.hz
    try:
        while not system.viewer.quit:
            t_iter_start = time.monotonic()
            _datapoints, _scene_pcd, _robot_pcd, _robot_link_pcds = system.step()
            if period_s is not None:
                elapsed = time.monotonic() - t_iter_start
                time.sleep(max(0.0, period_s - elapsed))
    finally:
        system.close()


if __name__ == "__main__":
    main()
