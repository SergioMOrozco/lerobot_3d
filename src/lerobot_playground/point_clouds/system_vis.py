import foxglove
import logging
import os
import json
import shutil
import threading
import time
from numbers import Real
from collections.abc import Mapping, Sequence

import open3d as o3d
import numpy as np
import imageio
import cv2

import foxglove
from foxglove.schemas import FrameTransforms
from foxglove.schemas import PointCloud, PackedElementField, PackedElementFieldNumericType
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot_playground.hardware_config import TeleopSystemConfig
from lerobot_playground.paths import CALIBRATION_DIR
from lerobot_playground.point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from lerobot_playground.point_clouds.point_cloud_viewer import LivePointCloudViewer
from lerobot_playground.point_clouds.robot_state import RobotState
from lerobot_playground.point_clouds.tuner import StateTuner
from foxglove.schemas import Pose, Vector3, Quaternion

def foxglove_pointcloud_from_numpy(points: np.ndarray, colors=None):

    N = points.shape[0]
    assert points.shape[1] == 3

    # Default alpha = 255
    if colors is None:
        colors = np.zeros((N, 3), dtype=np.uint8)

    # Ensure uint8
    colors = colors.astype(np.uint8)

    # Add alpha channel (uint8 = 255)
    a = np.full((N,1), 255, dtype=np.uint8)

    # Build structured array matching Foxglove's fields
    structured = np.zeros(N, dtype=[
        ("x", "float32"),
        ("y", "float32"),
        ("z", "float32"),
        ("b", "uint8"),
        ("g", "uint8"),
        ("r", "uint8"),
        ("a", "uint8"),
    ])

    structured["x"] = points[:,0]
    structured["y"] = points[:,1]
    structured["z"] = points[:,2]

    structured["r"] = colors[:,0]
    structured["g"] = colors[:,1]
    structured["b"] = colors[:,2]
    structured["a"] = 255

    data = structured.tobytes()

    fields = [
        PackedElementField(name="x", offset=0,  type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="y", offset=4,  type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="z", offset=8,  type=PackedElementFieldNumericType.Float32),
        PackedElementField(name="red", offset=12, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="green", offset=13, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="blue", offset=14, type=PackedElementFieldNumericType.Uint8),
        PackedElementField(name="alpha", offset=15, type=PackedElementFieldNumericType.Uint8),
    ]

    identity_pose = Pose(
        position=Vector3(x=0.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    )

    return PointCloud(
        timestamp=None,
        frame_id="base_link",
        pose=identity_pose,
        point_stride=16,
        fields=fields,
        data=data
    )

class SystemStateViewer:
    def __init__(
        self,
        config: TeleopSystemConfig,
    ):
        self.publish_to_foxglove = config.publish_to_foxglove
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
        self.pcd_viewer = (
            LivePointCloudViewer(point_size=config.point_size)
            if config.display_point_cloud_viewer
            else None
        )
        self.mask_provider = config.mask_provider

        serials = list(config.realsense_serials)
        self.stream = MultiRealSenseStream(
            serials,
            config.extrinsic_json,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
        )
        self.followers = [
            SO101Follower(SO101FollowerConfig(port=ax.port, id=ax.id)) for ax in config.followers
        ]

        self.recording_name = config.recording_name

        if self.recording_name != '':
            self.record = True
        else:
            self.record = False

        self.quit=False

        self.state_tuner: StateTuner | None = None
        if config.tune:
            self.state_tuner = StateTuner()
            self.state_tuner.start()

        print("Connecting robots...")
        for bot in self.followers:
            bot.connect()
        print("Connected.")
        if self.action_interpolation_duration_s > 0:
            self._start_action_thread()

        urdf = config.urdf_path or str(CALIBRATION_DIR / "so101_new_calib.urdf")
        # FK / mesh visualization uses the first follower's observation and its calibration id.
        calibration_path = (
            config.robot_calibration_paths[0]
            if config.robot_calibration_paths is not None
            else None
        )
        self.robot_state = RobotState(
            urdf,
            config.robot_calibration_ids[0],
            calibration_dir=config.robot_calibration_dir,
            calibration_path=calibration_path,
        )

        if self.publish_to_foxglove:
            foxglove.set_log_level(logging.INFO)
            foxglove.start_server()

        self.serials = serials
        self.images = {}
        self.depths = {}
        self.robot_pcds = []

        for serial in serials:
            self.images[serial] = []
            self.depths[serial] = []

    def update(self, *actions, masks_by_serial=None):
        # actions are simply joint states
        #
        # masks_by_serial can be either a mapping {serial: mask} or a sequence
        # aligned with self.serials/datapoints. Nonzero/True mask pixels are kept.
        #
        # Returns:
        #     scene_pcd: ``(N, 3)`` float64 world points from the fused scene cloud.
        #     robot_pcd: ``(M, 3)`` float64 world points for the sampled follower mesh.
        #     robot_link_pcds: per-link robot point clouds keyed by URDF link name.

        if self.state_tuner is not None and self.state_tuner.quit is True:
            self.quit = True

        if len(actions) != len(self.followers):
            raise ValueError(
                f"Expected {len(self.followers)} leader actions, got {len(actions)}"
            )
        self._set_action_targets(actions)

        with self._follower_io_lock:
            obs = self.followers[0].get_observation()

        transforms, robot_pcd_np, robot_link_pcds = self.robot_state.get_transforms(obs)
        datapoints = self.stream.get_datapoints()
        if masks_by_serial is None and self.mask_provider is not None:
            masks_by_serial = self.mask_provider(datapoints)
        self._apply_masks(datapoints, masks_by_serial)

        if self.state_tuner is not None and self.state_tuner.capture:
            self.state_tuner.capture = False

            calibration_dir = "calibration_files"

            # remove task directory if it exists
            if os.path.exists(calibration_dir):
                shutil.rmtree(calibration_dir)

            os.makedirs(calibration_dir)

            for datapoint in datapoints:

                serial_dir = os.path.join(calibration_dir, datapoint['serial'])

                # remove task directory if it exists
                if os.path.exists(serial_dir):
                    shutil.rmtree(serial_dir)

                os.makedirs(serial_dir)

                cv2.imwrite(os.path.join(serial_dir, "color.png"), datapoint['color'])
                np.savez_compressed(os.path.join(serial_dir, "depth.npz"), depth=np.array(datapoint['depth']))
            np.savez_compressed(os.path.join(calibration_dir, "robot_pcd.npz"), pcd=np.array(robot_pcd_np))

        if self.record:
            for datapoint in datapoints:
                self.images[datapoint['serial']].append(np.array(datapoint['color']))
                self.depths[datapoint['serial']].append(np.array(datapoint['depth']))
            self.robot_pcds.append(np.array(robot_pcd_np))

        scene_pcd, pcd_list = get_fused_point_cloud(
           datapoints
        )

        st = getattr(self, "state_tuner", None)
        if st is not None and st.save_subgoal:
           st.save_subgoal = False
           self._save_scene_pcd_subgoal(scene_pcd)

        scene_pcd_np = np.asarray(scene_pcd.points, dtype=np.float64)
        robot_pcd_np = np.asarray(robot_pcd_np, dtype=np.float64)

        if self.publish_to_foxglove:
            for idx, pcd in enumerate(pcd_list):
                pts = np.asarray(pcd.points, dtype=np.float32)
                cols = np.asarray(np.array(pcd.colors) * 255, dtype=np.uint8) if pcd.has_colors() else None
                pcd_msg = foxglove_pointcloud_from_numpy(pts, cols)
                foxglove.log(f"/pcd_{idx}", pcd_msg)

            robot_pcd_msg = foxglove_pointcloud_from_numpy(robot_pcd_np.astype(np.float32, copy=False))
            foxglove.log("/robot_pcd", robot_pcd_msg)
            foxglove.log(
               "/tf",
               FrameTransforms(transforms=transforms)
            )

        if self.pcd_viewer is not None:
            scene_colors = (
                np.asarray(scene_pcd.colors, dtype=np.float64)
                if scene_pcd.has_colors()
                else np.full((scene_pcd_np.shape[0], 3), 0.7, dtype=np.float64)
            )
            robot_colors = np.tile(np.array([[1.0, 0.1, 0.1]], dtype=np.float64), (robot_pcd_np.shape[0], 1))
            viewer_points = np.vstack((scene_pcd_np, robot_pcd_np))
            viewer_colors = np.vstack((scene_colors, robot_colors))
            self.pcd_viewer.update(viewer_points, viewer_colors)

        return scene_pcd_np, robot_pcd_np, robot_link_pcds


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
        actions = []
        for start_action, target_action in zip(self._start_actions, self._target_actions):
            out = {}
            for key, target_value in target_action.items():
                start_value = start_action.get(key, target_value)
                if isinstance(start_value, Real) and isinstance(target_value, Real):
                    out[key] = float(start_value) + alpha * (float(target_value) - float(start_value))
                else:
                    out[key] = target_value
            actions.append(out)
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
            masks = [masks_by_serial.get(datapoint["serial"]) for datapoint in datapoints]
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
                datapoint["obj_mask"] = None
                continue
            mask_np = np.asarray(mask)
            if mask_np.shape[:2] != datapoint["depth"].shape[:2]:
                raise ValueError(
                    f"Mask for camera {datapoint['serial']} has shape {mask_np.shape}; "
                    f"expected {datapoint['depth'].shape[:2]}"
                )
            datapoint["obj_mask"] = mask_np


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

        if self.record:

            recording_dir = f"recordings/{self.recording_name}"

            # remove task directory if it exists
            if os.path.exists(recording_dir):
                shutil.rmtree(recording_dir)

            os.makedirs(recording_dir)

            for serial in self.serials:

                serial_dir = os.path.join(recording_dir, f"{serial}" )

                # remove task directory if it exists
                if os.path.exists(serial_dir):
                    shutil.rmtree(serial_dir)

                os.makedirs(serial_dir)

                frames_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images[serial]]
                frames_depth = self.depths[serial]

                imageio.mimsave(
                    os.path.join(serial_dir, "rgb.mp4"),
                    frames_rgb,
                    fps=30,
                    codec="libx264"
                )

                np.savez_compressed(os.path.join(serial_dir, "depth.npz"), depth=np.array(frames_depth))
            np.savez_compressed(os.path.join(recording_dir, "robot_pcd.npz"), pcd=np.array(self.robot_pcds))

        if not os.path.exists("intrinsic_calibration.json"):
            datapoints = self.stream.get_datapoints()

            intrinsics = {}
            for datapoint in datapoints:

                intr = datapoint["color_intrinsics"]

                intrinsics[datapoint['serial']] = {}
                intrinsics[datapoint['serial']]['fl_x'] = intr.fx
                intrinsics[datapoint['serial']]['fl_y'] = intr.fy
                intrinsics[datapoint['serial']]['cx'] = intr.ppx
                intrinsics[datapoint['serial']]['cy'] = intr.ppy
                intrinsics[datapoint['serial']]['w'] = datapoint['color'].shape[1]
                intrinsics[datapoint['serial']]['h'] = datapoint['color'].shape[0]

            with open("intrinsic_calibration.json", "w") as f:
                json.dump(intrinsics, f, indent=8)

        if self.pcd_viewer is not None:
            self.pcd_viewer.close()

        self.stream.stop()