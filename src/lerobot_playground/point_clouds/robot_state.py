"""SO101 follower URDF state: FK and mesh sampling for visualization."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from scipy.spatial.transform import Rotation
from urchin import URDF

from lerobot_playground.paths import CALIBRATION_DIR


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

        # Cache sampled visual geometry and full mesh geometry in each link frame.
        # Runtime only applies FK transforms.
        self.link_visual_points: list[tuple[str, np.ndarray]] = []
        self.link_visual_meshes: list[tuple[str, np.ndarray, np.ndarray]] = []
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

    def ticks_to_radians(self, raw):
        """Tick reading -> radians relative to the robot's home pose.

        ``raw`` (e.g. calibration range_min/range_max, or a live Present_Position
        read) is already homing-corrected by the motor's own firmware --
        Feetech servos compute ``Present_Position = Actual_Position -
        Homing_Offset`` on-device, so every tick value LeRobot's software ever
        sees already has that applied. The reference point application code
        needs is the fixed encoder half-turn (half of TICKS_PER_REV - 1), which
        is what LeRobot's ``set_half_turn_homings`` calibration procedure
        defines as the robot's home/zero pose -- not ``homing_offset`` itself,
        which is firmware bookkeeping, not something to re-apply here.
        """
        TICKS_PER_REV = 4096
        HALF_TURN = (TICKS_PER_REV - 1) // 2
        return (raw - HALF_TURN) * (2 * np.pi / TICKS_PER_REV)

    def compute_phys_ranges(self, calib_dict):
        """This robot's own calibrated physical range per joint (radians), from its
        calibration file's range_min/range_max -- see
        convert_lerobot_action_to_radians for why this must be per-robot, not a
        generic reference range."""
        phys_ranges = {}

        for joint, data in calib_dict.items():
            lo = self.ticks_to_radians(data["range_min"])
            hi = self.ticks_to_radians(data["range_max"])

            phys_ranges[joint] = {
                "lo": float(lo),
                "hi": float(hi),
                "drive_mode": bool(data.get("drive_mode", 0)),
            }

        return phys_ranges

    def load_robot_meshes(self):
        """Load meshes once in each link frame: full geometry for URDF rendering,
        plus a 100-point sample for the lightweight point-cloud representation."""
        for link in self.robot_urdf.links:
            for visual in link.visuals:
                if not hasattr(visual.geometry, "mesh"):
                    continue

                mesh_path = (CALIBRATION_DIR / visual.geometry.mesh.filename).resolve()
                mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))

                if mesh_o3d.is_empty():
                    print(f"[WARN] Empty mesh: {mesh_path}")
                    continue

                T_vis = visual.origin
                R_vis = T_vis[:3, :3]
                t_vis = T_vis[:3, 3]

                verts_mesh = np.asarray(mesh_o3d.vertices)
                verts_visual = (R_vis @ verts_mesh.T).T + t_vis
                faces = np.asarray(mesh_o3d.triangles)
                self.link_visual_meshes.append(
                    (link.name, verts_visual.astype(np.float64, copy=False), faces)
                )

                pts_mesh = np.asarray(mesh_o3d.sample_points_uniformly(100).points)
                pts_visual = (R_vis @ pts_mesh.T).T + t_vis
                self.link_visual_points.append((link.name, pts_visual.astype(np.float64, copy=False)))

    def convert_lerobot_action_to_radians(self, joint_state):
        """Un-normalize LeRobot's motor-space observation into this robot's own
        calibrated physical joint angles (radians), via self.PHYS_RANGES.

        obs["<joint>.pos"] is produced by MotorsBus._normalize as a fraction of
        *this specific motor's* calibrated range_min/range_max (RANGE_M100_100 for
        all joints, RANGE_0_100 for the gripper -- see lerobot.motors.motors_bus).
        It has no relationship to any generic/reference angular range, so it must
        be un-normalized through this robot's own calibration, not a shared table
        -- two physical units calibrated at different times can have meaningfully
        different range_min/range_max.
        """
        positions = {}
        for joint, r in self.PHYS_RANGES.items():
            lo_clip = 0.0 if joint == "gripper" else -100.0
            norm = float(np.clip(joint_state[f"{joint}.pos"], lo_clip, 100.0))
            if joint == "gripper":
                frac = (100.0 - norm if r["drive_mode"] else norm) / 100.0
            else:
                frac = ((100.0 - norm) if r["drive_mode"] else (100.0 + norm)) / 200.0
            positions[joint] = r["lo"] + frac * (r["hi"] - r["lo"])

        return positions

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

    def get_static_meshes(self):
        """Local-frame (rest-pose) ``(link_name, mesh_name, vertices, faces)`` per visual mesh.

        Call this once at startup, not per frame. Meant to be mounted under a
        per-link frame node in the viewer and moved rigidly via that frame's
        pose (see :func:`get_link_poses`) -- these meshes have tens of thousands
        of vertices each, so re-transforming and re-uploading them every frame
        (instead of just moving a frame) is the dominant per-frame cost to avoid.
        """
        return [
            (link_name, f"{link_name}_{i}", verts_visual, faces)
            for i, (link_name, verts_visual, faces) in enumerate(self.link_visual_meshes)
        ]

    def get_link_poses(self, fk_poses):
        """World-frame ``(translation, quaternion_wxyz)`` per link with visual meshes.

        Deliberately cheap: only a 3-vector + quaternion per link, not mesh
        vertices -- see :func:`get_static_meshes`.
        """
        poses = {}
        for link_name in {name for name, _, _ in self.link_visual_meshes}:
            T_link = fk_poses[self.robot_urdf.link_map[link_name]]
            translation = T_link[:3, 3].astype(np.float64, copy=False)
            quat_xyzw = Rotation.from_matrix(T_link[:3, :3]).as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
            )
            poses[link_name] = (translation, quat_wxyz)
        return poses

    def get_eef_pos(self, obs):
        joint_positions = self.get_joint_positions(obs)

        return self.robot_urdf.link_fk(cfg=joint_positions)[self.robot_urdf.link_map["gripper_frame_link"]]

    def get_robot_state(self, obs):
        """FK the current observation; return world-frame point clouds and per-link poses."""
        joint_positions = self.convert_lerobot_action_to_radians(obs)
        fk_poses = self.robot_urdf.link_fk(cfg=joint_positions)
        robot_pcd, robot_link_pcds = self.sample_robot_points(fk_poses)
        link_poses = self.get_link_poses(fk_poses)
        return robot_pcd, robot_link_pcds, link_poses
