"""SO101 follower URDF state: FK, Foxglove transforms, and mesh sampling for visualization."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from foxglove.schemas import FrameTransform, Quaternion, Vector3
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from urchin import URDF

from lerobot_playground.paths import CALIBRATION_DIR


# joint limit written in USD (degree)
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan.pos": (-110.0, 110.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 90.0),
    "wrist_flex.pos": (-95.0, 95.0),
    "wrist_roll.pos": (-160.0, 160.0),
    "gripper.pos": (-10, 100.0),
}

# motor limit written in real device (normalized to related range)
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan.pos": (-100.0, 100.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 100.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-100.0, 100.0),
    "gripper.pos": (0.0, 100.0),
}


class RobotState:
    def __init__(
        self,
        urdf_path,
        id,
        *,
        robot_type: str = "so101_follower",
        calibration_dir: str | Path | None = None,
        calibration_path: str | Path | None = None,
    ):
        self.robot_urdf = URDF.load(urdf_path)

        robot_calibration_path = self._resolve_calibration_path(
            id,
            robot_type=robot_type,
            calibration_dir=calibration_dir,
            calibration_path=calibration_path,
        )

        with open(robot_calibration_path, "r") as f:
            calib = json.load(f)

        self.PHYS_RANGES = self.compute_phys_ranges(calib)

        # Cache sampled visual geometry in each link frame. Runtime only applies FK transforms.
        self.link_visual_points: list[tuple[str, np.ndarray]] = []
        self.load_robot_meshes()

    def _resolve_calibration_path(
        self,
        id: str,
        *,
        robot_type: str,
        calibration_dir: str | Path | None,
        calibration_path: str | Path | None,
    ) -> Path:
        if calibration_path is not None:
            path = Path(calibration_path).expanduser()
        else:
            root = (
                Path(calibration_dir).expanduser()
                if calibration_dir is not None
                else HF_LEROBOT_CALIBRATION / ROBOTS / robot_type
            )
            path = root / f"{id}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"Robot calibration file not found for id '{id}': {path}. "
                "Pass TeleopSystemConfig.robot_calibration_dir / robot_calibration_paths, "
                "or set HF_LEROBOT_CALIBRATION so LeRobot and lerobot_playground use the same files."
            )
        return path

    def ticks_to_radians(self, raw, homing_offset):
        TICKS_PER_REV = 4096
        return (raw + homing_offset) * (2 * np.pi / TICKS_PER_REV)

    def compute_phys_ranges(self, calib_dict):
        phys_ranges = {}

        for joint, data in calib_dict.items():
            range_min = data["range_min"]
            range_max = data["range_max"]
            offset = data["homing_offset"]

            lo = self.ticks_to_radians(range_min, offset)
            hi = self.ticks_to_radians(range_max, offset)

            phys_ranges[joint] = [float(lo), float(hi)]

        return phys_ranges

    def rot_matrix_to_quat(self, R):
        """
        Convert a 3x3 rotation matrix to quaternion [x, y, z, w].
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([x, y, z, w], dtype=np.float64)

    def load_robot_meshes(self):
        """Load meshes and sample visual points once in each link frame."""
        for link in self.robot_urdf.links:
            for visual in link.visuals:
                if not hasattr(visual.geometry, "mesh"):
                    continue

                mesh_path = (CALIBRATION_DIR / visual.geometry.mesh.filename).resolve()
                mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))

                if mesh_o3d.is_empty():
                    print(f"[WARN] Empty mesh: {mesh_path}")
                    continue

                pts_mesh = np.asarray(mesh_o3d.sample_points_uniformly(100).points)

                T_vis = visual.origin
                R_vis = T_vis[:3, :3]
                t_vis = T_vis[:3, 3]
                pts_visual = (R_vis @ pts_mesh.T).T + t_vis
                self.link_visual_points.append((link.name, pts_visual.astype(np.float64, copy=False)))

    def convert_lerobot_action_to_radians(self, joint_state):
        """
        Convert the action from Lerobot to LeIsaac. Just convert value, not include the format.
        """

        processed_action = np.zeros(6)
        joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
        motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

        for idx, joint_name in enumerate(joint_limits):
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = joint_limits[joint_name]
            motor_range = motor_limit_range[1] - motor_limit_range[0]
            joint_range = joint_limit_range[1] - joint_limit_range[0]
            motor_degree = joint_state[joint_name] - motor_limit_range[0]
            processed_degree = motor_degree / motor_range * joint_range + joint_limit_range[0]
            processed_radius = processed_degree / 180.0 * np.pi  # convert to radian
            processed_action[idx] = processed_radius

        return processed_action

    def get_joint_positions(self, obs):
        """Joint configuration for FK (radians)."""
        return self.convert_lerobot_action_to_radians(obs)

    def sample_robot_points(self, fk_poses):
        """Return full and per-link robot point clouds from cached link-frame samples."""
        per_link_parts: dict[str, list[np.ndarray]] = defaultdict(list)

        for link_name, pts_visual in self.link_visual_points:
            T_link = fk_poses[self.robot_urdf.link_map[link_name]]
            R_link, t_link = T_link[:3, :3], T_link[:3, 3]
            pts_world = (R_link @ pts_visual.T).T + t_link
            per_link_parts[link_name].append(pts_world)

        per_link = {
            link_name: np.vstack(parts).astype(np.float64, copy=False)
            for link_name, parts in per_link_parts.items()
        }
        full = (
            np.vstack(list(per_link.values())).astype(np.float64, copy=False)
            if per_link
            else np.empty((0, 3), dtype=np.float64)
        )
        return full, per_link

    def get_eef_pos(self, obs):
        joint_positions = self.get_joint_positions(obs)

        return self.robot_urdf.link_fk(cfg=joint_positions)[self.robot_urdf.link_map["gripper_frame_link"]]

    def get_transforms(self, obs):

        transforms = []

        joint_positions = self.convert_lerobot_action_to_radians(obs)

        # Compute forward kinematics with updated joint positions
        fk_poses = self.robot_urdf.link_fk(cfg=joint_positions)

        # World -> Base
        transforms.append(
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="base",
                translation=Vector3(x=0.0, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )

        for joint in self.robot_urdf.joints:
            parent_link = joint.parent
            child_link = joint.child
            T_parent = fk_poses[self.robot_urdf.link_map[parent_link]]
            T_child = fk_poses[self.robot_urdf.link_map[child_link]]

            # Local transform from parent->child
            T_local = np.linalg.inv(T_parent) @ T_child
            trans = T_local[:3, 3]
            quat = self.rot_matrix_to_quat(T_local[:3, :3])
            transforms.append(
                FrameTransform(
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    translation=Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                    rotation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
                )
            )

        robot_pcd, robot_link_pcds = self.sample_robot_points(fk_poses)
        return transforms, robot_pcd, robot_link_pcds
