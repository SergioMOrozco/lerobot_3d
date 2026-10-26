import os
import json
import math
import shutil
import threading
import time
from numbers import Real
from collections.abc import Mapping, Sequence

import open3d as o3d
import numpy as np
import cv2

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.utils.constants import ROBOTS, TELEOPERATORS
from lerobot_3d.teleop_config import TeleopSystemConfig
from lerobot_3d.paths import CALIBRATION_DIR, YAM_URDF_PATH
from lerobot_3d.point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from lerobot_3d.point_clouds.viser_viewer import CameraView, ViserSceneViewer, grid_offsets
from lerobot_3d.point_clouds.robot_state import RobotState
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState, yam_base_poses
from lerobot_3d.calibration_mask import (
    SEGMENT_HELP,
    RobotSegmenter,
    save_mask_png,
    segment_help,
    segment_interactively,
)
from lerobot_3d.table_plane import (
    TABLE_PLANE_KEY,
    fit_table_plane,
    load_table_plane,
    masked_world_points,
)

_HOMING_DURATION_S = 3.0
"""Seconds to move the followers back to their startup pose before segmenting a capture."""
_RESUME_DURATION_S = 2.0
"""Seconds to blend from the home pose back to the teleop target once segmentation is done."""


def write_intrinsics(datapoints, path="intrinsic_calibration.json") -> None:
    """Add/refresh each datapoint's color intrinsics in ``path``, keeping other cameras' entries."""
    intrinsics = {}
    if os.path.exists(path):
        with open(path) as f:
            intrinsics = json.load(f)
    for datapoint in datapoints:
        intr = datapoint.color_intrinsics
        intrinsics[datapoint.serial] = {
            'fl_x': intr.fx,
            'fl_y': intr.fy,
            'cx': intr.ppx,
            'cy': intr.ppy,
            'w': datapoint.color.shape[1],
            'h': datapoint.color.shape[0],
        }
    with open(path, "w") as f:
        json.dump(intrinsics, f, indent=8)


def camera_views(datapoints) -> dict[str, CameraView]:
    """Each datapoint's camera as drawn in viser: pose, field of view and its color image (RGB)."""
    views = {}
    for datapoint in datapoints:
        if datapoint.X_WC is None or datapoint.color_intrinsics is None:
            continue
        height, width = (datapoint.color if datapoint.color is not None else datapoint.depth).shape[:2]
        views[datapoint.serial] = CameraView(
            X_WC=np.asarray(datapoint.X_WC, dtype=np.float64),
            fov=2.0 * math.atan2(height / 2.0, float(datapoint.color_intrinsics.fy)),
            aspect=width / height,
            # Datapoint color is BGR (RealSense bgr8 layout); viser wants RGB.
            image=None if datapoint.color is None else np.ascontiguousarray(datapoint.color[..., ::-1]),
        )
    return views


def blend_actions(start_actions, target_actions, alpha):
    """Per-key linear blend of motor-space action dicts; non-numeric values take the target."""
    actions = []
    for start_action, target_action in zip(start_actions, target_actions):
        out = {}
        for key, target_value in target_action.items():
            start_value = start_action.get(key, target_value)
            if isinstance(start_value, Real) and isinstance(target_value, Real):
                out[key] = float(start_value) + alpha * (float(target_value) - float(start_value))
            else:
                out[key] = target_value
        actions.append(out)
    return actions


class SystemStateViewer:
    def __init__(
        self,
        config: TeleopSystemConfig,
    ):
        self.action_interpolation_duration_s = config.action_interpolation_duration_s
        self.action_command_hz = config.action_command_hz
        self._action_lock = threading.Lock()
        self._follower_io_lock = threading.Lock()
        self._action_stop_event = threading.Event()
        self._action_thread: threading.Thread | None = None
        self._current_actions = None
        self._target_actions = None
        self._start_actions = None
        self._target_start_time = 0.0
        self.viewer = ViserSceneViewer(
            point_size=config.point_size, port=config.viser_port, controls=config.tune
        )

        serials = list(config.realsense_serials)
        # No cameras -> no stream (and no extrinsics file needed); the scene cloud stays empty.
        self.stream = (
            MultiRealSenseStream(
                serials,
                config.extrinsic_json,
                width=config.camera_width,
                height=config.camera_height,
                fps=config.camera_fps,
            )
            if serials
            else None
        )
        self.robot_type = config.robot_type
        if config.robot_type == "yam":
            # Imported here so SO101 setups never touch ROS.
            from lerobot_3d.robots.yam_follower import YamFollower, YamFollowerConfig

            self.followers = [
                YamFollower(YamFollowerConfig(side=arm.side, id=f"yam_{arm.side}", joint_topic=arm.joint_topic,
                                              command_topic=arm.command_topic))
                for arm in config.followers
            ]
        else:
            self.followers = [
                SO101Follower(SO101FollowerConfig(port=ax.port, id=ax.id)) for ax in config.followers
            ]
        self.yam_followers = config.robot_type == "yam" and bool(self.followers)
        """Real YAM arms: every one is read each update, and commanded when ``update()`` gets actions."""
        self.sent_actions: list[dict] | None = None
        """What the YAM followers were actually sent in the last ``update()`` (after their speed limits)."""

        self.segment_on_capture = config.segment_on_capture
        self._segmenter = RobotSegmenter(config.sam2_model_id)
        self._home_actions = None
        """Follower motor positions at startup; captures home here before segmenting."""
        self._resume_from = None
        """``(actions, start_time)`` to blend teleop targets from after homing, else ``None``."""

        self.quit=False

        if self.followers:
            print("Connecting robots...")
            for bot in self.followers:
                bot.connect()
            print("Connected.")
            if not self.yam_followers:
                self._home_actions = self._read_follower_positions()
                if self.action_interpolation_duration_s > 0:
                    self._start_action_thread()
        else:
            print("No followers configured; actions only pose the URDF in viser.")

        if config.robot_type == "yam":
            self._init_yam_robots(config)
        else:
            self._init_so101_robots(config)
        self.serials = serials

        self.extrinsic_json = config.extrinsic_json
        self.table_plane = load_table_plane(config.extrinsic_json)
        """The fitted table (``table_plane`` in the extrinsics file), or ``None`` until it's fit."""
        if self.table_plane is not None:
            self.viewer.draw_table_plane(self.table_plane)

        self.ros_stream = None
        self._link_robot: dict[str, int] = {}
        """URDF link -> index of the robot it belongs to, for placing linked ROS cameras."""
        self._last_link_poses: dict[str, np.ndarray] | None = None
        if config.ros_cameras:
            from lerobot_3d.point_clouds.ros_camera_stream import RosCameraStream

            self.ros_stream = RosCameraStream(config.ros_cameras, config.extrinsic_json)
            sides = [state.side for state in self.robot_states]
            self._link_robot = {link: sides.index(link.split("_")[0]) for link in self.ros_stream.links}

    def _init_yam_robots(self, config: TeleopSystemConfig) -> None:
        # One YamRobotState per arm; each snapshot carries the arm's station pose (the URDF's, or a
        # calibrated yam_<side>_base from the extrinsics file), so there is no grid layout.
        from lerobot_3d.point_clouds.ros_camera_stream import extrinsics_path, read_extrinsics

        urdf = config.urdf_path or str(YAM_URDF_PATH)
        self.base_poses = yam_base_poses(read_extrinsics(extrinsics_path(config.extrinsic_json)))
        """Calibrated arm base poses in use, ``{side: X_W_base}``; arms without one use the URDF's."""
        for side, pose in self.base_poses.items():
            if side in config.yam_sides:
                print(f"YAM {side} arm base: calibrated pose at {np.round(pose[:3, 3], 3)} m")
        self.robot_states = [YamRobotState(urdf, side, base_pose=self.base_poses.get(side)) for side in config.yam_sides]
        self.num_robots = len(self.robot_states)
        self.base_offsets = [None] * self.num_robots
        for i, robot_state in enumerate(self.robot_states):
            self.viewer.load_static_meshes(robot_state.get_static_meshes(), robot_index=i)
        self.robot_state = self.robot_states[0]

    def rest_actions(self) -> list[dict[str, float]]:
        """One rest-pose action per robot (YAM only): poses virtual arms with no hardware."""
        return [robot_state.rest_observation() for robot_state in self.robot_states]

    def _init_so101_robots(self, config: TeleopSystemConfig) -> None:
        urdf = config.urdf_path or str(CALIBRATION_DIR / "so101_new_calib.urdf")
        # With followers, FK / mesh visualization uses the first follower's observation and its
        # calibration id. With no followers, num_robots virtual robots are each posed by their
        # own commanded action and drawn on a grid.
        self.num_robots = 1 if self.followers else config.num_robots
        self.base_offsets = grid_offsets(self.num_robots, config.robot_grid_spacing)
        # Robots sharing a calibration share one RobotState (URDF meshes are loaded once).
        states_by_calibration: dict[tuple, RobotState] = {}
        self.robot_states: list[RobotState] = []
        for i in range(self.num_robots):
            calibration_id = config.robot_calibration_ids[i]
            calibration_path = (
                config.robot_calibration_paths[i]
                if config.robot_calibration_paths is not None
                else None
            )
            key = (calibration_id, calibration_path)
            if key not in states_by_calibration:
                states_by_calibration[key] = RobotState(
                    urdf,
                    calibration_id,
                    robot_type=config.calibration_robot_type,
                    calibration_category=(
                        TELEOPERATORS
                        if config.calibration_robot_type == "so101_leader"
                        else ROBOTS
                    ),
                    calibration_dir=config.robot_calibration_dir,
                    calibration_path=calibration_path,
                )
            robot_state = states_by_calibration[key]
            self.robot_states.append(robot_state)
            self.viewer.load_static_meshes(
                robot_state.get_static_meshes(), robot_index=i, base_offset=self.base_offsets[i]
            )
        self.robot_state = self.robot_states[0]

    def update(self, *actions, masks_by_serial=None):
        # actions are simply joint states: one per follower, or one per virtual robot
        # (num_robots) with no followers -- those pose the URDFs directly instead of being
        # sent to hardware.
        #
        # masks_by_serial can be either a mapping {serial: mask} or a sequence
        # aligned with self.serials/datapoints. Nonzero/True mask pixels are kept.
        #
        # Returns:
        #     datapoints: raw per-camera datapoints used to build the fused point cloud.
        #     scene_pcd: Open3D point cloud with fused scene points and colors.
        #     robot_pcds: per robot, ``(M, 3)`` float64 sampled mesh points (robot base frame).
        #     robot_link_pcds: per robot, point clouds keyed by URDF link name (base frame).
        #     robot_snapshots: per robot, a RobotSnapshot (joint state, clouds, link poses,
        #         grid base_offset).

        if self.viewer.quit:
            self.quit = True

        if self.yam_followers:
            # No actions: just observe. Actions: command every arm, then observe.
            if actions and len(actions) != len(self.followers):
                raise ValueError(f"Expected {len(self.followers)} actions (one per YAM arm), got {len(actions)}")
            self.sent_actions = [f.send_action(a) for f, a in zip(self.followers, actions)] if actions else None
            observations = [follower.get_observation() for follower in self.followers]
        elif len(actions) != (expected := len(self.followers) if self.followers else self.num_robots):
            raise ValueError(f"Expected {expected} actions, got {len(actions)}")
        elif self.followers:
            self._set_action_targets(self._blend_from_resume(actions))
            with self._follower_io_lock:
                observations = [self.followers[0].get_observation()]
        else:
            observations = [dict(action) for action in actions]

        snapshots = [
            robot_state.get_robot_snapshot(obs, index=i, base_offset=self.base_offsets[i])
            for i, (robot_state, obs) in enumerate(zip(self.robot_states, observations))
        ]
        datapoints = self.stream.get_datapoints() if self.stream is not None else []
        link_poses = None
        if self.ros_stream is not None:
            link_poses = {
                link: self.robot_states[i].get_link_transform(observations[i], link)
                for link, i in self._link_robot.items()
            }
            self._last_link_poses = link_poses
            datapoints += self.ros_stream.get_datapoints(link_poses)

        joint_positions = {key: value for obs in observations for key, value in obs.items()}
        for datapoint in datapoints:
            datapoint.joint_positions = joint_positions

        if self.viewer.capture:
            self.viewer.capture = False

            calibration_dir = "calibration_files"

            # remove task directory if it exists
            if os.path.exists(calibration_dir):
                shutil.rmtree(calibration_dir)

            os.makedirs(calibration_dir)

            for datapoint in datapoints:

                serial_dir = os.path.join(calibration_dir, datapoint.serial)

                # remove task directory if it exists
                if os.path.exists(serial_dir):
                    shutil.rmtree(serial_dir)

                os.makedirs(serial_dir)

                cv2.imwrite(os.path.join(serial_dir, "color.png"), datapoint.color)
                np.savez_compressed(os.path.join(serial_dir, "depth.npz"), depth=np.array(datapoint.depth))
            np.savez_compressed(
                os.path.join(calibration_dir, "robot_pcd.npz"),
                pcd=np.array(self._capture_robot_pcd(observations, snapshots)),
            )
            if self.robot_type == "yam":
                self._write_capture_info(calibration_dir, observations)
            # icp.py needs intrinsics for every captured camera; don't wait for shutdown.
            write_intrinsics(datapoints)
            if self.ros_stream is not None:
                self._write_extrinsic_seeds(self.ros_stream.extrinsic_seeds(link_poses))

            if self.segment_on_capture:
                self._home_followers()
                self._segment_capture(datapoints, calibration_dir)

        if self.viewer.fit_table:
            self.viewer.fit_table = False
            self._fit_table(datapoints)

        scene_pcd, _ = get_fused_point_cloud(
           datapoints
        )

        if self.viewer.save_subgoal:
           self.viewer.save_subgoal = False
           self._save_scene_pcd_subgoal(scene_pcd)

        scene_pcd_np = np.asarray(scene_pcd.points, dtype=np.float64)

        scene_colors = (
            np.asarray(scene_pcd.colors, dtype=np.float64)
            if scene_pcd.has_colors()
            else None
        )
        self.viewer.update(scene_pcd_np, scene_colors, snapshots)
        if datapoints:
            self.viewer.update_cameras(camera_views(datapoints))

        return (
            datapoints,
            scene_pcd,
            [snapshot.pcd for snapshot in snapshots],
            [snapshot.link_pcds for snapshot in snapshots],
            snapshots,
        )


    def _start_action_thread(self) -> None:
        self._action_thread = threading.Thread(target=self._action_loop, daemon=True)
        self._action_thread.start()


    def _copy_actions(self, actions):
        return [dict(action) for action in actions]


    def _set_action_targets(self, actions) -> None:
        actions = self._copy_actions(actions)
        if self.action_interpolation_duration_s <= 0:
            self._send_actions(actions)
            return

        send_immediately = False
        with self._action_lock:
            if self._current_actions is None:
                self._current_actions = self._copy_actions(actions)
                self._start_actions = self._copy_actions(actions)
                self._target_actions = self._copy_actions(actions)
                send_immediately = True
            else:
                self._start_actions = self._copy_actions(self._current_actions)
                self._target_actions = self._copy_actions(actions)
            self._target_start_time = time.monotonic()

        if send_immediately:
            self._send_actions(actions)


    def _action_loop(self) -> None:
        period_s = 1.0 / self.action_command_hz
        while not self._action_stop_event.is_set():
            t0 = time.monotonic()
            with self._action_lock:
                actions = self._interpolated_actions_locked(t0)
            if actions is not None:
                self._send_actions(actions)
            elapsed = time.monotonic() - t0
            self._action_stop_event.wait(max(0.0, period_s - elapsed))


    def _interpolated_actions_locked(self, now):
        if self._target_actions is None:
            return None
        duration = self.action_interpolation_duration_s
        alpha = min(1.0, max(0.0, (now - self._target_start_time) / duration))
        actions = blend_actions(self._start_actions, self._target_actions, alpha)
        self._current_actions = self._copy_actions(actions)
        return actions


    def _send_actions(self, actions) -> None:
        with self._follower_io_lock:
            for follower, action in zip(self.followers, actions):
                follower.send_action(action)


    def _apply_masks(self, datapoints, masks_by_serial) -> None:
        if masks_by_serial is None:
            return

        if isinstance(masks_by_serial, Mapping):
            masks = [masks_by_serial.get(datapoint.serial) for datapoint in datapoints]
        elif isinstance(masks_by_serial, Sequence) and not isinstance(masks_by_serial, (str, bytes)):
            if len(masks_by_serial) != len(datapoints):
                raise ValueError(
                    f"Expected {len(datapoints)} masks, got {len(masks_by_serial)}"
                )
            masks = list(masks_by_serial)
        else:
            raise TypeError("masks_by_serial must be a mapping, sequence, or None")

        for datapoint, mask in zip(datapoints, masks):
            if mask is None:
                datapoint.obj_mask = None
                continue
            mask_np = np.asarray(mask)
            if mask_np.shape[:2] != datapoint.depth.shape[:2]:
                raise ValueError(
                    f"Mask for camera {datapoint.serial} has shape {mask_np.shape}; "
                    f"expected {datapoint.depth.shape[:2]}"
                )
            datapoint.obj_mask = mask_np


    def _read_follower_positions(self):
        with self._follower_io_lock:
            observations = [follower.get_observation() for follower in self.followers]
        return [
            {key: float(value) for key, value in obs.items() if key.endswith(".pos")}
            for obs in observations
        ]

    def _home_followers(self) -> None:
        """Blocking move of the followers back to their startup pose, held until resume."""
        if not self.followers or self._home_actions is None:
            return
        print("[capture] Homing followers to their startup pose...")
        start = self._read_follower_positions()
        # Pause the interpolation thread so it doesn't fight the homing move.
        with self._action_lock:
            self._target_actions = None
        period_s = 1.0 / self.action_command_hz
        t0 = time.monotonic()
        while True:
            alpha = min(1.0, (time.monotonic() - t0) / _HOMING_DURATION_S)
            self._send_actions(blend_actions(start, self._home_actions, alpha))
            if alpha >= 1.0:
                break
            time.sleep(period_s)
        with self._action_lock:
            if self._current_actions is not None:
                # Interpolation thread resumes by holding the home pose.
                self._current_actions = self._copy_actions(self._home_actions)
                self._start_actions = self._copy_actions(self._home_actions)
                self._target_actions = self._copy_actions(self._home_actions)
        self._resume_from = (self._copy_actions(self._home_actions), None)

    def _blend_from_resume(self, actions):
        """After homing, ease teleop targets from the home pose instead of jumping to them."""
        if self._resume_from is None:
            return actions
        start, t0 = self._resume_from
        now = time.monotonic()
        if t0 is None:
            # Clock starts on the first teleop tick after segmentation, not when homing ended.
            t0 = now
            self._resume_from = (start, t0)
        alpha = min(1.0, (now - t0) / _RESUME_DURATION_S)
        if alpha >= 1.0:
            self._resume_from = None
            return actions
        return blend_actions(start, actions, alpha)

    def _capture_robot_pcd(self, observations, snapshots) -> np.ndarray:
        """World-frame ICP target. SO101: robot 0 sits at the world origin, so its base-frame cloud
        is already world frame. YAM: every arm, densely sampled at its station pose."""
        if self.robot_type == "yam":
            return np.concatenate([
                robot_state.get_mesh_points(obs) for robot_state, obs in zip(self.robot_states, observations)
            ])
        return snapshots[0].pcd

    def _write_capture_info(self, calibration_dir: str, observations) -> None:
        """``capture.json``: which arms are in robot_pcd.npz and their base poses, so icp.py can also
        keep each fixed camera's pose relative to an arm it was calibrated against alone."""
        info = {
            "robot_type": "yam",
            "arms": {
                state.side: state.get_link_transform(obs, f"{state.side}_base").tolist()
                for state, obs in zip(self.robot_states, observations)
            },
            "linked_cameras": sorted(self.ros_stream.mounts) if self.ros_stream is not None else [],
        }
        with open(os.path.join(calibration_dir, "capture.json"), "w") as f:
            json.dump(info, f, indent=4)
        print(f"[capture] ICP target: {' + '.join(info['arms'])} arm(s); cameras: "
              f"{', '.join(os.path.basename(d) for d in sorted(os.listdir(calibration_dir)) if os.path.isdir(os.path.join(calibration_dir, d)))}")

    def _write_extrinsic_seeds(self, seeds: dict) -> None:
        """Add/replace linked cameras' entries (starting ``X_WC`` plus ``link``/``X_WL``) in the
        extrinsics file, keeping every other camera's entry."""
        if not seeds:
            return
        path = self.ros_stream.extrinsics_file
        extrinsics = {}
        if os.path.exists(path):
            with open(path) as f:
                extrinsics = json.load(f)
        extrinsics.update(seeds)
        with open(path, "w") as f:
            json.dump(extrinsics, f, indent=8)

    def _segment_capture(self, datapoints, calibration_dir: str) -> None:
        # Blocks the teleop loop on purpose: the robot has to hold the captured pose anyway
        # (robot_pcd.npz), and followers keep their last commanded target meanwhile.
        print(f"[capture] Segment the robot in each camera's window. {SEGMENT_HELP}")
        for datapoint in datapoints:
            try:
                mask = segment_interactively(
                    self._segmenter, datapoint.color, f"Segment robot - {datapoint.serial}"
                )
            except ImportError as e:
                print(f"[capture] {e} Skipping mask.png for all cameras.")
                return
            mask_path = os.path.join(calibration_dir, datapoint.serial, "mask.png")
            if mask is None:
                print(f"[capture] Skipped {datapoint.serial}: no mask.png, so icp.py will skip it.")
                continue
            save_mask_png(mask_path, datapoint.color, mask)
            print(f"[capture] Wrote {mask_path}")

    def _fit_table(self, datapoints) -> None:
        """Click-segment the table in each camera, fit one plane to all their world points, save it as
        the extrinsics file's ``table_plane`` entry and draw it. Blocks the loop like Capture's masks."""
        from lerobot_3d.point_clouds.ros_camera_stream import extrinsics_path, read_extrinsics

        print(f"[table] Segment the table in each camera's window. {segment_help('table')}")
        clouds = {}
        for datapoint in datapoints:
            if datapoint.color is None or datapoint.X_WC is None:
                continue
            try:
                mask = segment_interactively(
                    self._segmenter, datapoint.color, f"Segment table - {datapoint.serial}", target="table"
                )
            except ImportError as e:
                print(f"[table] {e}")
                return
            if mask is None:
                print(f"[table] Skipped {datapoint.serial}.")
                continue
            intr = datapoint.color_intrinsics
            K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
            clouds[datapoint.serial] = masked_world_points(
                datapoint.depth, mask, K, datapoint.depth_scale, datapoint.X_WC, max_depth=datapoint.max_depth
            )
        points = np.concatenate(list(clouds.values())) if clouds else np.zeros((0, 3))
        try:
            plane = fit_table_plane(points)
        except ValueError as e:
            print(f"[table] Not saved: {e}")
            return
        print(f"[table] Plane: normal {np.round(plane.normal, 4)}, table top at "
              f"{np.round(plane.X_W_table[:3, 3], 4)} m, {plane.extent[0]:.3f} x {plane.extent[1]:.3f} m, "
              f"rms {plane.rms * 1000:.1f} mm over {plane.inliers} points")
        # Each camera's median height above the fused plane: one far from 0 has extrinsic error.
        for serial, cloud in clouds.items():
            heights = plane.signed_distance(cloud)
            heights = heights[np.abs(heights) < 0.03]
            if len(heights):
                print(f"[table]   {serial}: median offset {np.median(heights) * 1000:+.1f} mm")
        path = extrinsics_path(self.extrinsic_json)
        extrinsics = read_extrinsics(path)
        extrinsics[TABLE_PLANE_KEY] = plane.to_json()
        with open(path, "w") as f:
            json.dump(extrinsics, f, indent=8)
        print(f"[table] Wrote {TABLE_PLANE_KEY} to {path}")
        self.table_plane = plane
        self.viewer.draw_table_plane(plane)

    def _save_scene_pcd_subgoal(self, scene_pcd: o3d.geometry.PointCloud) -> None:
        subgoals_dir = "subgoals"
        os.makedirs(subgoals_dir, exist_ok=True)
        next_idx = 1
        if os.path.isdir(subgoals_dir):
            for name in os.listdir(subgoals_dir):
                if name.endswith(".npz") and name[:-4].isdigit():
                    next_idx = max(next_idx, int(name[:-4]) + 1)
        path = os.path.join(subgoals_dir, f"{next_idx}.npz")
        pts = np.asarray(scene_pcd.points, dtype=np.float32)
        payload: dict = {"pts": pts}
        if scene_pcd.has_colors():
            payload["colors"] = np.asarray(scene_pcd.colors, dtype=np.float32)
        np.savez_compressed(path, **payload)
        print(f"[SystemStateViewer] Saved fused scene to {path} ({pts.shape[0]} points)")

    def close(self):
        self._action_stop_event.set()
        if self._action_thread is not None:
            self._action_thread.join(timeout=1.0)

        if self.stream is not None:
            write_intrinsics(self.stream.get_datapoints())
        if self.ros_stream is not None and self._last_link_poses is not None:
            write_intrinsics(self.ros_stream.get_datapoints(self._last_link_poses))

        self.viewer.close()

        if self.stream is not None:
            self.stream.stop()
        if self.robot_type == "yam":
            for follower in self.followers:
                follower.disconnect()
        if self.ros_stream is not None or self.robot_type == "yam":
            from lerobot_3d.ros_hub import shutdown_ros_hub

            shutdown_ros_hub()