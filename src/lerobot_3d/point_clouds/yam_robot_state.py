"""YAM v1 / linear_4310 station URDF adapter; no hardware access."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from urchin import URDF

from lerobot_3d.common.types import RobotSnapshot


def transform_points(matrix, points):
    return np.einsum("ij,nj->ni", matrix[:3, :3], points) + matrix[:3, 3]


LIMIT_SLACK = (0.2, 0.1)
"""Beyond-limit slack (radians for arm joints, fraction of full opening for the gripper) treated as
sensor noise and clipped. Rest sits exactly on some lower limits (joints 2 and 3 at 0), so measured
noise lands just past them. Anything further out means the joint mapping is wrong and raises."""


def base_pose_key(side: str) -> str:
    """Extrinsics-file entry holding a calibrated arm base pose, ``{"X_W_base": 4x4}``."""
    return f"yam_{side}_base"


def yam_base_poses(extrinsics) -> dict[str, np.ndarray]:
    """``{side: X_W_base}`` for every arm with a calibrated base pose in an extrinsics dict."""
    return {
        side: np.asarray(extrinsics[base_pose_key(side)]["X_W_base"], dtype=np.float64)
        for side in ("left", "right")
        if base_pose_key(side) in extrinsics
    }


def base_correction(urdf_base: np.ndarray, base_pose: np.ndarray | None) -> np.ndarray:
    """World transform moving an arm from its URDF base pose to ``base_pose`` (identity if ``None``)."""
    if base_pose is None:
        return np.eye(4)
    return np.asarray(base_pose, dtype=np.float64) @ np.linalg.inv(urdf_base)


def visual_color(visual, default=(200, 200, 200)) -> tuple[int, int, int]:
    """A URDF visual's material color as 0-255 RGB (``default`` without one)."""
    material = visual.material
    if material is None or material.color is None:
        return default
    return tuple(int(round(float(c) * 255)) for c in np.asarray(material.color)[:3])


def joint_keys(side: str) -> list[str]:
    """One arm's LeRobot-style observation/action keys, in URDF joint order."""
    return [f"{side}_joint{i}.pos" for i in range(1, 7)] + [f"{side}_gripper.pos"]


class YamRobotState:
    """FK and mesh sampling for one arm of I2RT's composed YAM station URDF.

    Observations are keyed like a LeRobot robot (see :func:`joint_keys`): ``<side>_joint1..6.pos``
    in radians and ``<side>_gripper.pos`` as the gripper opening, 0 closed to 1 fully open (the YAM's own
    ``<side>_gripper`` command convention), which drives both linear_4310 fingers.

    Base and camera placements are those in the URDF, so snapshots are placed at the arm's station
    pose rather than on a grid. ``base_pose`` (a calibrated world pose of ``<side>_base``, see
    :func:`yam_base_poses`) moves the whole arm, its links and wrist camera included, from the URDF's
    base pose to it.
    """

    def __init__(self, urdf_path, side, base_pose: np.ndarray | None = None):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.side = side
        self.keys = joint_keys(side)
        self.robot_urdf = URDF.load(str(urdf_path))
        self.joints = [self.robot_urdf.joint_map[f"{side}_joint{i}"] for i in range(1, 9)]
        for i, joint in enumerate(self.joints):
            expected = "revolute" if i < 6 else "prismatic"
            if joint.joint_type != expected or joint.limit is None:
                raise ValueError("Expected the composed YAM v1 / linear_4310 station URDF")
            if i >= 6 and (joint.limit.lower != 0 or joint.limit.upper <= 0):
                raise ValueError("Expected positive-travel linear_4310 fingers")
        self._meshes, self._points, self._mesh_colors = [], [], []
        for link in self.robot_urdf.links:
            if not link.name.startswith(side + "_"):
                continue
            for visual_index, visual in enumerate(link.visuals):
                geometry = visual.geometry.mesh
                if geometry is None:
                    continue
                for mesh_index, original in enumerate(geometry.meshes):
                    mesh = original.copy()
                    if geometry.scale is not None:
                        mesh.apply_scale(geometry.scale)
                    mesh.apply_transform(visual.origin)
                    self._meshes.append(
                        (link.name, f"{visual_index}_{mesh_index}", mesh.vertices, mesh.faces)
                    )
                    self._mesh_colors.append(visual_color(visual))
                    self._points.append((link.name, trimesh.sample.sample_surface(mesh, 500)[0]))
        if not self._meshes:
            raise ValueError("The selected arm has no visual meshes")
        self._calibration_points = None
        urdf_base = self.robot_urdf.link_fk(use_names=True)[side + "_base"]
        self.correction = base_correction(urdf_base, base_pose)
        """Applied in front of every URDF link pose of this arm (identity without ``base_pose``)."""

    def rest_observation(self) -> dict[str, float]:
        """The URDF's own rest configuration: every joint at 0, fingers closed."""
        return dict.fromkeys(self.keys, 0.0)

    def get_joint_positions(self, observation):
        """Observation -> URDF joint values, with sensor noise at the limits clipped away."""
        try:
            values = [float(observation[key]) for key in self.keys]
        except KeyError as e:
            raise ValueError(f"YAM {self.side} observation is missing {e}") from e
        out = {}
        for joint, value in zip(self.joints[:6], values[:6]):
            lower, upper = joint.limit.lower, joint.limit.upper
            if not np.isfinite(value) or not lower - LIMIT_SLACK[0] <= value <= upper + LIMIT_SLACK[0]:
                raise ValueError(
                    f"{joint.name}={value} is outside the URDF limits [{lower}, {upper}]; check the joint mapping"
                )
            out[joint.name] = float(np.clip(value, lower, upper))
        opening = values[6]
        if not np.isfinite(opening) or not -LIMIT_SLACK[1] <= opening <= 1 + LIMIT_SLACK[1]:
            raise ValueError(f"{self.side}_gripper={opening} is outside the 0 (closed) .. 1 (open) range")
        for joint in self.joints[6:]:  # both fingers travel together
            out[joint.name] = float(np.clip(opening, 0.0, 1.0) * joint.limit.upper)
        return out

    def get_static_meshes(self):
        """``(link, mesh name, vertices, faces, rgb)`` per visual mesh; rgb is the URDF material's color."""
        return [(*mesh, color) for mesh, color in zip(self._meshes, self._mesh_colors)]

    def get_link_transform(self, observation, link):
        """World (URDF) pose of ``link`` at ``observation``."""
        return self.correction @ self.robot_urdf.link_fk(cfg=self.get_joint_positions(observation), use_names=True)[link]

    def get_robot_snapshot(self, observation, index=0, base_offset=None):
        """Same interface as ``RobotState.get_robot_snapshot``; ``base_offset`` is ignored because
        the arm's base pose comes from the station URDF."""
        joints = self.get_joint_positions(observation)
        poses = self.robot_urdf.link_fk(cfg=joints, use_names=True)
        local = {name: np.linalg.inv(poses[self.side + "_base"]) @ matrix for name, matrix in poses.items()}
        base = self.correction @ poses[self.side + "_base"]  # link poses stay base-relative
        parts, part_colors = defaultdict(list), defaultdict(list)
        for (link, points), color in zip(self._points, self._mesh_colors):
            parts[link].append(transform_points(local[link], points))
            part_colors[link].append(np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1)))
        per_link = {name: np.concatenate(points) for name, points in parts.items()}
        per_link_colors = {name: np.concatenate(colors) for name, colors in part_colors.items()}
        link_poses = {
            name: (local[name][:3, 3], Rotation.from_matrix(local[name][:3, :3]).as_quat()[[3, 0, 1, 2]])
            for name in per_link
        }
        return RobotSnapshot(
            index=index,
            joint_positions={key: float(observation[key]) for key in self.keys},
            joint_radians={joint.name: joints[joint.name] for joint in self.joints[:6]},
            pcd=np.concatenate(list(per_link.values())),
            link_pcds=per_link,
            link_poses=link_poses,
            base_offset=base[:3, 3],
            base_wxyz=tuple(Rotation.from_matrix(base[:3, :3]).as_quat()[[3, 0, 1, 2]]),
            pcd_colors=np.concatenate(list(per_link_colors.values())),
            link_pcd_colors=per_link_colors,
        )

    def get_mesh_points(self, observation):
        """Dense mesh samples of this arm in the URDF world frame: the ICP target on Capture."""
        # ICP's final 2 mm correspondence radius needs a denser target than the
        # lightweight display cloud. Cache these independent samples once.
        if self._calibration_points is None:
            self._calibration_points = [
                (
                    name,
                    trimesh.sample.sample_surface(
                        trimesh.Trimesh(vertices=vertices, faces=faces, process=False), 10000
                    )[0],
                )
                for name, _, vertices, faces in self._meshes
            ]
        poses = self.robot_urdf.link_fk(cfg=self.get_joint_positions(observation), use_names=True)
        return np.concatenate([
            transform_points(self.correction @ poses[name], points) for name, points in self._calibration_points
        ])
