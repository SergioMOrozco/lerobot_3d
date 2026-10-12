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


class YamRobotState:
    """Use I2RT's composed station URDF and measured radians, without SO101 scaling.

    ``position_rad`` contains six arm angles. ``gripper_open`` is normalized
    closed=0/open=1 for the station's two positive-travel linear_4310 fingers.
    Base and camera placements are exactly those in the supplied URDF.
    """

    def __init__(self, urdf_path, side):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.side = side
        self.robot_urdf = URDF.load(str(urdf_path))
        self.joints = [self.robot_urdf.joint_map[f"{side}_joint{i}"] for i in range(1, 9)]
        for i, joint in enumerate(self.joints):
            expected = "revolute" if i < 6 else "prismatic"
            if joint.joint_type != expected or joint.limit is None:
                raise ValueError("Expected the composed YAM v1 / linear_4310 station URDF")
            if i >= 6 and (joint.limit.lower != 0 or joint.limit.upper <= 0):
                raise ValueError("Expected positive-travel linear_4310 fingers")
        self._meshes, self._points = [], []
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
                    self._points.append((link.name, trimesh.sample.sample_surface(mesh, 500)[0]))
        if not self._meshes:
            raise ValueError("The selected arm has no visual meshes")
        self._calibration_points = None

    def get_joint_positions(self, observation):
        q = np.asarray(observation["position_rad"], dtype=float)
        opening = float(observation["gripper_open"])
        if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError("Expected six finite radians and gripper_open in [0, 1]")
        values = [*q, *(opening * joint.limit.upper for joint in self.joints[6:])]
        for joint, value in zip(self.joints, values):
            if not joint.limit.lower - 1e-5 <= value <= joint.limit.upper + 1e-5:
                raise ValueError(f"{joint.name} outside URDF limits; verify measured joint mapping")
        return {joint.name: float(value) for joint, value in zip(self.joints, values)}

    def get_static_meshes(self):
        return self._meshes

    def get_link_transform(self, observation, link):
        return self.robot_urdf.link_fk(cfg=self.get_joint_positions(observation), use_names=True)[link]

    def get_robot_snapshot(self, observation, index=0):
        joints = self.get_joint_positions(observation)
        poses = self.robot_urdf.link_fk(cfg=joints, use_names=True)
        base = poses[self.side + "_base"]
        local = {name: np.linalg.inv(base) @ matrix for name, matrix in poses.items()}
        parts = defaultdict(list)
        for link, points in self._points:
            parts[link].append(transform_points(local[link], points))
        per_link = {name: np.concatenate(points) for name, points in parts.items()}
        link_poses = {
            name: (local[name][:3, 3], Rotation.from_matrix(local[name][:3, :3]).as_quat()[[3, 0, 1, 2]])
            for name in per_link
        }
        return RobotSnapshot(
            index=index,
            joint_positions=joints,
            joint_radians={joint.name: joints[joint.name] for joint in self.joints[:6]},
            pcd=np.concatenate(list(per_link.values())),
            link_pcds=per_link,
            link_poses=link_poses,
            base_offset=base[:3, 3],
            base_wxyz=tuple(Rotation.from_matrix(base[:3, :3]).as_quat()[[3, 0, 1, 2]]),
        )

    def get_mesh_points(self, observation, *, gripper_only=False):
        """Mesh target in URDF world, or just this gripper in its own flange frame.

        Gripper-local calibration needs only ``gripper_open``: arm joint angles
        cancel between the camera mount and its own gripper. Camera meshes are
        excluded from this target. Dimensions and finger travel come from URDF.
        """
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
        if gripper_only:
            # An arbitrary internal FK reference, never represented as feedback.
            observation = {"position_rad": np.zeros(6), "gripper_open": observation["gripper_open"]}
        poses = self.robot_urdf.link_fk(cfg=self.get_joint_positions(observation), use_names=True)
        reference = np.linalg.inv(poses[self.side + "_gripper"]) if gripper_only else np.eye(4)
        gripper_links = {self.side + suffix for suffix in ("_gripper", "_tip_left", "_tip_right")}
        return np.concatenate(
            [
                transform_points(reference @ poses[name], points)
                for name, points in self._calibration_points
                if not gripper_only or name in gripper_links
            ]
        )
