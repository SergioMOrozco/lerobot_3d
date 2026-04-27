import foxglove
import logging
import os
import json
import shutil

import open3d as o3d
import numpy as np
import imageio
import cv2

import foxglove
from foxglove.schemas import FrameTransforms
from foxglove.schemas import PointCloud, PackedElementField, PackedElementFieldNumericType
from lerobot_playground.hardware_config import TeleopSystemConfig
from lerobot_playground.paths import CALIBRATION_DIR
from lerobot_playground.point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from lerobot_playground.point_clouds.point_cloud_viewer import LivePointCloudViewer
from lerobot_playground.point_clouds.robot_state import RobotState
from lerobot_playground.point_clouds.tuner import StateTuner
from foxglove.schemas import Pose, Vector3, Quaternion
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

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
        self.pcd_viewer = (
            LivePointCloudViewer(point_size=config.point_size)
            if config.display_point_cloud_viewer
            else None
        )

        serials = list(config.realsense_serials)
        self.stream = MultiRealSenseStream(serials, config.extrinsic_json)
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

        urdf = config.urdf_path or str(CALIBRATION_DIR / "so101_new_calib.urdf")
        # FK / mesh visualization uses the first follower's observation and its calibration id.
        self.robot_state = RobotState(urdf, config.robot_calibration_ids[0])

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

    def update(self, *actions):
        # actions are simply joint states
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
        for follower, action in zip(self.followers, actions):
            follower.send_action(action)

        obs = self.followers[0].get_observation()

        transforms, robot_pcd_np, robot_link_pcds = self.robot_state.get_transforms(obs)
        datapoints = self.stream.get_datapoints()

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