"""Fast FK and IK for one YAM arm's end effector, from the station URDF's joint chain.

urchin's ``link_fk`` is far too slow to call inside an IK loop every control step, so this walks the
chain from the URDF root to ``<side>_gripper`` once and then composes plain 4x4 matrices. Only the six
revolute arm joints move the end effector; the fingers don't.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from urchin import URDF


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return T


class YamArmKinematics:
    """End-effector pose of one arm as a function of its six joint angles (world = URDF root)."""

    def __init__(self, urdf_path, side: str, ee_link: str | None = None):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.side = side
        self.ee_link = ee_link or f"{side}_gripper"
        urdf = URDF.load(str(urdf_path), lazy_load_meshes=True)
        by_child = {joint.child: joint for joint in urdf.joints}
        chain, link = [], self.ee_link
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].parent
        chain.reverse()
        arm = [f"{side}_joint{i}" for i in range(1, 7)]
        if [j.name for j in chain if j.joint_type != "fixed"] != arm:
            raise ValueError(f"Expected {arm} between the URDF root and {self.ee_link}")
        # Fixed joints before joint1 collapse into the arm's base pose.
        self.T_world_base = np.eye(4)
        first = next(i for i, j in enumerate(chain) if j.joint_type != "fixed")
        for joint in chain[:first]:
            self.T_world_base = self.T_world_base @ joint.origin
        self._joints = [(np.asarray(j.origin, dtype=np.float64), np.asarray(j.axis, dtype=np.float64),
                         j.joint_type != "fixed") for j in chain[first:]]
        moving = [j for j in chain[first:] if j.joint_type != "fixed"]
        self.lower = np.array([j.limit.lower for j in moving])
        self.upper = np.array([j.limit.upper for j in moving])
        self.finger_upper = float(urdf.joint_map[f"{side}_joint7"].limit.upper)
        """Full finger travel (metres)."""

    def _frames(self, q):
        """End-effector pose plus each moving joint's world axis and origin (for the Jacobian)."""
        T = self.T_world_base.copy()
        axes, origins, i = [], [], 0
        for origin, axis, moves in self._joints:
            T = T @ origin
            if moves:
                axes.append(T[:3, :3] @ axis)
                origins.append(T[:3, 3].copy())
                T = T @ _axis_angle(axis, q[i])
                i += 1
        return T, np.array(axes), np.array(origins)

    def fk(self, q) -> np.ndarray:
        """World pose of the end-effector link for joint angles ``q`` (6 radians)."""
        return self._frames(np.asarray(q, dtype=np.float64))[0]

    def ik(self, T_target, q0, *, rotation_weight=0.2, iterations=60, damping=0.02, tol=1e-5) -> np.ndarray:
        """Joint angles reaching ``T_target`` as closely as the arm and its limits allow.

        Damped least squares from ``q0`` (the current joints), so solutions stay continuous between
        steps. ``rotation_weight`` is metres of position error worth one radian of orientation error.
        Unreachable targets return the closest pose found, never one outside the joint limits.
        """
        q = np.clip(np.asarray(q0, dtype=np.float64), self.lower, self.upper)
        W = np.diag([1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight])
        for _ in range(iterations):
            T, axes, origins = self._frames(q)
            error = np.concatenate([
                T_target[:3, 3] - T[:3, 3],
                Rotation.from_matrix(T_target[:3, :3] @ T[:3, :3].T).as_rotvec(),
            ])
            if np.linalg.norm(W @ error) < tol:
                break
            J = np.vstack([np.cross(axes, T[:3, 3] - origins).T, axes.T])  # geometric Jacobian, 6x6
            Jw = W @ J
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping**2 * np.eye(6), W @ error)
            q = np.clip(q + dq, self.lower, self.upper)
        return q
