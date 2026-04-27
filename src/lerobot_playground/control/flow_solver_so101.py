#!/usr/bin/env python3
"""
Standalone demo of the Jacobian-based flow solver for SO101.

Two robots side by side in viser:
  - LEFT (blue):  Controlled by sliders (source gripper point cloud)
  - RIGHT (red):  Follows via point-cloud flow + Jacobian IK

**Recording** (this file — not ``flow_solver.py``):
  1. Move the left arm with sliders while **Record** is active; gripper PCs are saved to
     ``recorded_flow.npy`` when you stop.
  2. **Solve trajectory** runs IK; the solved joint path is exported to
     ``recorded_flow_robot.npz`` (motor-space, same format as ``flow_solver.py play``).
  3. Optional: **Load model joints** reads ``recorded_joints.npz`` (``arm_qpos``, ``gripper_openness``),
     regenerates gripper flow via FK, and writes ``recorded_flow.npy`` for solving.

Dependencies: urchin (URDF), numpy, viser, trimesh, open3d
"""
from __future__ import annotations

import os
import time
import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import viser
import viser.transforms as vtf
from urchin import URDF

from lerobot_playground.paths import CALIBRATION_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
URDF_PATH = str(CALIBRATION_DIR / "so101_new_calib.urdf")

# Flow / robot trajectory artifacts (default: current working directory).
_ARTIFACT_DIR = Path(os.environ.get("LEROBOT_PLAYGROUND_ARTIFACT_DIR", ".")).resolve()

# Arm joints in kinematic-chain order (base to tip)
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"
EEF_LINK_NAME = "gripper_link"

# Gripper links whose point clouds we track
GRIPPER_COLLISION_LINKS = ["gripper_link", "moving_jaw_so101_v1_link"]

# Openness -> drive_val mapping (from config)
OPEN_RAD = 1.74533    # fully open
CLOSED_RAD = -0.174533  # fully closed

# Points per link for the flow PC
PTS_PER_LINK = 80

# Jacobian solver params
JAC_ITERS = 10
JAC_DAMPING = 1e-3
JAC_EPS = 5e-4

# Lateral offset to place robots side by side
OFFSET = np.array([0.0, 0.5, 0.0])

VISER_PORT = 7007


# ---------------------------------------------------------------------------
# Helpers: urdfpy-based FK + PC sampling
# ---------------------------------------------------------------------------

def trimesh_to_o3d(tmesh):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
    m.triangles = o3d.utility.Vector3iVector(tmesh.faces)
    return m


class URDFRobot:
    """Lightweight wrapper around urdfpy for FK + point-cloud sampling."""

    def __init__(self, urdf_path: str, origin: np.ndarray = np.zeros(3), shared_local_pts: dict[str, np.ndarray] | None = None):
        self.robot = URDF.load(urdf_path)
        self.origin = np.array(origin, dtype=np.float64)

        # Pre-load collision meshes for the gripper links
        self._collision_meshes: dict[str, o3d.geometry.TriangleMesh] = {}
        self._collision_offsets: dict[str, np.ndarray] = {}
        for link in self.robot.links:
            if link.name not in GRIPPER_COLLISION_LINKS:
                continue
            if not link.collisions:
                continue
            col = link.collisions[0]
            if col.geometry.mesh is None or len(col.geometry.mesh.meshes) == 0:
                continue
            tmesh = col.geometry.mesh.meshes[0]
            scale = col.geometry.mesh.scale[0] if col.geometry.mesh.scale is not None else 1.0
            o3d_mesh = trimesh_to_o3d(tmesh)
            o3d_mesh.scale(scale, center=np.array([0, 0, 0]))
            self._collision_meshes[link.name] = o3d_mesh
            self._collision_offsets[link.name] = col.origin if col.origin is not None else np.eye(4)

        # Cache sampled points (in mesh-local frame) per link
        if shared_local_pts is not None:
            self._cached_local_pts = shared_local_pts
        else:
            self._cached_local_pts: dict[str, np.ndarray] = {}
            for name, mesh in self._collision_meshes.items():
                sampled = mesh.sample_points_poisson_disk(number_of_points=PTS_PER_LINK)
                self._cached_local_pts[name] = np.asarray(sampled.points)

        # Pre-load ALL visual meshes for full-robot display
        self._visual_meshes: dict[str, list] = {}   # link_name -> [(o3d_mesh, offset_4x4)]
        for link in self.robot.links:
            if not link.visuals:
                continue
            entries = []
            for vis in link.visuals:
                if vis.geometry.mesh is None or len(vis.geometry.mesh.meshes) == 0:
                    continue
                tmesh = vis.geometry.mesh.meshes[0]
                scale = vis.geometry.mesh.scale[0] if vis.geometry.mesh.scale is not None else 1.0
                o3d_mesh = trimesh_to_o3d(tmesh)
                o3d_mesh.scale(scale, center=np.array([0, 0, 0]))
                o3d_mesh.compute_vertex_normals()
                offset = vis.origin if vis.origin is not None else np.eye(4)
                entries.append((o3d_mesh, offset))
            if entries:
                self._visual_meshes[link.name] = entries

        # Joint state
        self.arm_qpos = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float64)
        self.gripper_openness = 1.0  # [0, 1]

    def _build_cfg(self, arm_qpos=None, gripper_openness=None):
        """Build urdfpy config dict from arm angles + gripper openness."""
        if arm_qpos is None:
            arm_qpos = self.arm_qpos
        if gripper_openness is None:
            gripper_openness = self.gripper_openness
        cfg = {}
        for i, name in enumerate(ARM_JOINT_NAMES):
            cfg[name] = float(arm_qpos[i])
        drive_val = gripper_openness * OPEN_RAD + (1.0 - gripper_openness) * CLOSED_RAD
        cfg[GRIPPER_JOINT_NAME] = drive_val
        return cfg

    def fk(self, arm_qpos=None, gripper_openness=None):
        """Run FK, return dict of link_name -> 4x4 world pose (with origin offset)."""
        cfg = self._build_cfg(arm_qpos, gripper_openness)
        link_fk = self.robot.link_fk(cfg=cfg)
        result = {}
        offset_mat = np.eye(4)
        offset_mat[:3, 3] = self.origin
        for link_obj, pose in link_fk.items():
            result[link_obj.name] = offset_mat @ pose
        return result

    def gripper_pc(self, arm_qpos=None, gripper_openness=None) -> np.ndarray:
        """Sample gripper point cloud in world frame. Returns (N, 3)."""
        poses = self.fk(arm_qpos, gripper_openness)
        all_pts = []
        for link_name in GRIPPER_COLLISION_LINKS:
            if link_name not in self._cached_local_pts:
                continue
            link_pose = poses[link_name]
            col_offset = self._collision_offsets[link_name]
            world_mat = link_pose @ col_offset
            local_pts = self._cached_local_pts[link_name]
            world_pts = local_pts @ world_mat[:3, :3].T + world_mat[:3, 3]
            all_pts.append(world_pts)
        return np.concatenate(all_pts, axis=0)

    def eef_pose(self, arm_qpos=None, gripper_openness=None) -> np.ndarray:
        """Return 4x4 EEF pose in world frame."""
        poses = self.fk(arm_qpos, gripper_openness)
        return poses[EEF_LINK_NAME]

    def get_visual_meshes_world(self, arm_qpos=None, gripper_openness=None):
        """Return list of (vertices, triangles) in world frame for all visual meshes."""
        poses = self.fk(arm_qpos, gripper_openness)
        result = []
        for link_name, entries in self._visual_meshes.items():
            if link_name not in poses:
                continue
            link_pose = poses[link_name]
            for o3d_mesh, offset in entries:
                world_mat = link_pose @ offset
                verts = np.asarray(o3d_mesh.vertices).copy()
                verts = verts @ world_mat[:3, :3].T + world_mat[:3, 3]
                tris = np.asarray(o3d_mesh.triangles)
                result.append((verts, tris))
        return result


# ---------------------------------------------------------------------------
# Jacobian IK solver (point-cloud based)
# ---------------------------------------------------------------------------

class JacobianFlowSolver:
    """
    Given a target gripper point cloud, solve for joint angles via
    damped-least-squares on the numerical Jacobian of the PC.
    """

    def __init__(self, robot: URDFRobot, n_iters=JAC_ITERS, damping=JAC_DAMPING, eps=JAC_EPS):
        self.robot = robot
        self.n_iters = n_iters
        self.damping = damping
        self.eps = eps
        self.n_arm = len(ARM_JOINT_NAMES)

    def _forward(self, arm_qpos, openness):
        return self.robot.gripper_pc(arm_qpos, openness)

    def _numerical_jacobian(self, arm_qpos, openness):
        n_params = self.n_arm + 1
        pc0 = self._forward(arm_qpos, openness)
        n_flat = pc0.size
        J = np.zeros((n_flat, n_params), dtype=np.float64)

        params = np.concatenate([arm_qpos, [openness]])
        for i in range(n_params):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += self.eps
            p_minus[i] -= self.eps
            if i == n_params - 1:  # clip openness
                p_plus[i] = np.clip(p_plus[i], 0.0, 1.0)
                p_minus[i] = np.clip(p_minus[i], 0.0, 1.0)
            pc_plus = self._forward(p_plus[:self.n_arm], float(p_plus[self.n_arm]))
            pc_minus = self._forward(p_minus[:self.n_arm], float(p_minus[self.n_arm]))
            denom = p_plus[i] - p_minus[i]
            if abs(denom) < 1e-10:
                J[:, i] = 0.0
            else:
                J[:, i] = (pc_plus.flatten() - pc_minus.flatten()) / denom
        return J

    def solve(self, target_pc: np.ndarray):
        """
        Solve for arm_qpos + openness that makes the robot's gripper PC
        match target_pc as closely as possible.

        Mutates self.robot.arm_qpos and self.robot.gripper_openness in place.
        Returns the final gripper PC.
        """
        arm_qpos = self.robot.arm_qpos.copy()
        openness = self.robot.gripper_openness

        for it in range(self.n_iters):
            current_pc = self._forward(arm_qpos, openness)
            residual = (target_pc - current_pc).flatten().astype(np.float64)

            J = self._numerical_jacobian(arm_qpos, openness)

            JtJ = J.T @ J
            JtJ[np.diag_indices(JtJ.shape[0])] += self.damping
            Jtr = J.T @ residual
            dparams = np.linalg.solve(JtJ, Jtr)

            arm_qpos += dparams[:self.n_arm]
            openness = float(np.clip(openness + dparams[self.n_arm], 0.0, 1.0))

        self.robot.arm_qpos = arm_qpos
        self.robot.gripper_openness = openness
        return self._forward(arm_qpos, openness)


# ---------------------------------------------------------------------------
# Viser visualization
# ---------------------------------------------------------------------------

def upload_robot_meshes(server, robot: URDFRobot, prefix: str, color: tuple):
    """Upload robot visual meshes to viser scene."""
    mesh_data = robot.get_visual_meshes_world()
    for i, (verts, tris) in enumerate(mesh_data):
        server.scene.add_mesh_simple(
            name=f"{prefix}/mesh_{i}",
            vertices=verts.astype(np.float32),
            faces=tris.astype(np.uint32),
            color=color,
            flat_shading=True,
        )


def upload_point_cloud(server, name: str, pts: np.ndarray, color: tuple, size=0.005):
    colors = np.tile(np.array(color, dtype=np.uint8), (pts.shape[0], 1))
    server.scene.add_point_cloud(
        name=name,
        points=pts.astype(np.float32),
        colors=colors,
        point_size=size,
        point_shape="circle",
    )


# ---------------------------------------------------------------------------
# Flow visualization helpers
# ---------------------------------------------------------------------------

def draw_flow_lines(server, flow_traj: list[np.ndarray], name="/flow_lines"):
    """Draw rainbow line segments connecting consecutive flow timesteps."""
    import colorsys
    n_steps = len(flow_traj) - 1
    if n_steps < 1:
        return
    all_segments = []
    all_seg_colors = []
    # Subsample points for cleaner lines (every 4th point)
    n_pts_total = flow_traj[0].shape[0]
    sparse_idx = np.arange(0, n_pts_total, 4)
    for t in range(n_steps):
        prev_points = flow_traj[t][sparse_idx]
        next_points = flow_traj[t + 1][sparse_idx]
        step_segments = np.stack([prev_points, next_points], axis=1)
        all_segments.append(step_segments)
        h0 = 0.83 * t / max(n_steps, 1)
        h1 = 0.83 * (t + 1) / max(n_steps, 1)
        c0 = np.array([int(c * 255) for c in colorsys.hsv_to_rgb(h0, 0.9, 1.0)], dtype=np.uint8)
        c1 = np.array([int(c * 255) for c in colorsys.hsv_to_rgb(h1, 0.9, 1.0)], dtype=np.uint8)
        n_pts = step_segments.shape[0]
        seg_colors = np.stack([
            np.tile(c0, (n_pts, 1)),
            np.tile(c1, (n_pts, 1)),
        ], axis=1)
        all_seg_colors.append(seg_colors)
    all_segments = np.concatenate(all_segments, axis=0)
    all_seg_colors = np.concatenate(all_seg_colors, axis=0)
    server.scene.add_line_segments(
        name=name,
        points=all_segments.astype(np.float32),
        colors=all_seg_colors,
        line_width=1.5,
    )


FLOW_SAVE_PATH = _ARTIFACT_DIR / "recorded_flow.npy"
ROBOT_TRAJ_SAVE_PATH = _ARTIFACT_DIR / "recorded_flow_robot.npz"
JOINT_SAVE_PATH = _ARTIFACT_DIR / "recorded_joints.npz"

# Motor-space keys / limits — inverse of ``RobotState.convert_lerobot_action_to_radians``
# in ``lerobot_playground.point_clouds.robot_state`` (USD limits in degrees).
SO101_FOLLOWER_USD_JOINT_LIMITS = {
    "shoulder_pan.pos": (-110.0, 110.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 90.0),
    "wrist_flex.pos": (-95.0, 95.0),
    "wrist_roll.pos": (-160.0, 160.0),
    "gripper.pos": (-10.0, 100.0),
}
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan.pos": (-100.0, 100.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 100.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-100.0, 100.0),
    "gripper.pos": (0.0, 100.0),
}
MOTOR_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def joint_radians_to_motor(joint_key: str, rad: float) -> float:
    """Map a single joint angle (rad) to lerobot motor command using USD + motor ranges."""
    ml = SO101_FOLLOWER_MOTOR_LIMITS[joint_key]
    jl = SO101_FOLLOWER_USD_JOINT_LIMITS[joint_key]
    processed_degree = float(rad * 180.0 / np.pi)
    motor_range = ml[1] - ml[0]
    joint_range = jl[1] - jl[0]
    if abs(joint_range) < 1e-12:
        return float(ml[0])
    return float(ml[0] + (processed_degree - jl[0]) / joint_range * motor_range)


def solved_state_to_motor_row(arm_qpos: np.ndarray, gripper_openness: float) -> np.ndarray:
    """Convert URDF arm radians + gripper openness [0,1] to one motor row (6,) for hardware."""
    row = np.empty(6, dtype=np.float64)
    for i, name in enumerate(ARM_JOINT_NAMES):
        row[i] = joint_radians_to_motor(f"{name}.pos", float(arm_qpos[i]))
    drive_rad = gripper_openness * OPEN_RAD + (1.0 - gripper_openness) * CLOSED_RAD
    row[5] = joint_radians_to_motor("gripper.pos", drive_rad)
    return row


def export_solved_joints_for_robot(
    solved_joints: list[tuple[np.ndarray, float]],
    path: Path,
    nominal_hz: float = 15.0,
) -> None:
    """Write ``motor`` (T, 6) + ``joint_keys`` for ``flow_solver.py play``."""
    if not solved_joints:
        return
    motor = np.stack([solved_state_to_motor_row(a, o) for a, o in solved_joints], axis=0)
    keys = np.array(MOTOR_JOINT_KEYS, dtype=object)
    np.savez_compressed(path, motor=motor, joint_keys=keys, hz=np.float64(nominal_hz))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading SO101 URDF...")
    source = URDFRobot(URDF_PATH, origin=-OFFSET / 2)
    target = URDFRobot(URDF_PATH, origin=OFFSET / 2, shared_local_pts=source._cached_local_pts)

    solver = JacobianFlowSolver(target)

    server = viser.ViserServer(host="0.0.0.0", port=VISER_PORT)
    print(f"Viser running at http://localhost:{VISER_PORT}")

    # --- Joint sliders for source robot ---
    arm_sliders = []
    for name in ARM_JOINT_NAMES:
        s = server.gui.add_slider(
            f"src/{name}", min=-3.14, max=3.14, step=0.01, initial_value=0.0,
        )
        arm_sliders.append(s)
    grip_slider = server.gui.add_slider(
        "src/gripper_openness", min=0.0, max=1.0, step=0.01, initial_value=1.0,
    )

    # --- Record / Solve controls ---
    record_button = server.gui.add_button("Record", color="green")
    solve_button = server.gui.add_button("Solve trajectory")
    solve_button.disabled = True
    load_joints_button = server.gui.add_button("Load model joints", color="blue")

    # Solver iteration slider
    iter_slider = server.gui.add_slider(
        "solver/n_iters", min=1, max=30, step=1, initial_value=JAC_ITERS,
    )

    # Playback slider (disabled until solved)
    playback_slider = server.gui.add_slider(
        "playback/timestep", min=0, max=1, step=1, initial_value=0,
    )
    playback_slider.disabled = True

    print("\nUse the sliders to move the LEFT (blue) robot.")
    print("Artifacts directory:", _ARTIFACT_DIR, "(override with env LEROBOT_PLAYGROUND_ARTIFACT_DIR)")
    print("Click 'Record' to record gripper point-cloud flow →", FLOW_SAVE_PATH.name)
    print("Click 'Solve trajectory' for IK → robot file →", ROBOT_TRAJ_SAVE_PATH.name)
    print("Then: lerobot-flow-solver play --input", ROBOT_TRAJ_SAVE_PATH.name)
    print("Or load joint keys from", JOINT_SAVE_PATH.name, "→ FK →", FLOW_SAVE_PATH.name)
    print("Scrub the playback slider to preview the solved target arm.\n")

    # --- State ---
    recording = False
    flow_traj: list[np.ndarray] = []  # list of (N_pts, 3) arrays
    solved_joints: list[tuple[np.ndarray, float]] = []  # list of (arm_qpos, openness)
    last_source_qpos = None

    @record_button.on_click
    def _on_record(_):
        nonlocal recording, flow_traj, solved_joints
        if not recording:
            # Start recording
            recording = True
            flow_traj = []
            solved_joints = []
            record_button.label = "Stop Recording"
            record_button.color = "red"
            solve_button.disabled = True
            playback_slider.disabled = True
            # Clear old visuals
            server.scene.remove_by_name("/flow_lines")
            server.scene.remove_by_name("/flow_lines_target")
            server.scene.remove_by_name("/target_grip_pc")
            print("[Record] Started")
        else:
            # Stop recording & save
            recording = False
            record_button.label = "Record"
            record_button.color = "green"
            if len(flow_traj) > 1:
                flow_arr = np.stack(flow_traj, axis=0)  # (T, N_pts, 3)
                np.save(str(FLOW_SAVE_PATH), flow_arr)
                print(f"[Record] Saved {flow_arr.shape[0]} frames to {FLOW_SAVE_PATH}")
                solve_button.disabled = False
            else:
                print("[Record] Too few frames, not saving")

    @load_joints_button.on_click
    def _on_load_joints(_):
        nonlocal flow_traj, solved_joints
        if not JOINT_SAVE_PATH.exists():
            print(f"[LoadJoints] No saved joints found at {JOINT_SAVE_PATH}")
            return

        data = np.load(str(JOINT_SAVE_PATH))
        arm_qpos_arr = data["arm_qpos"]
        grip_arr = data["gripper_openness"]
        n_frames = arm_qpos_arr.shape[0]
        print(f"[LoadJoints] Loaded {n_frames} frames, computing FK...")

        flow_traj = []
        for t in range(n_frames):
            pc = source.gripper_pc(arm_qpos_arr[t], float(grip_arr[t]))
            flow_traj.append(pc.copy())
            if (t + 1) % 100 == 0 or t == n_frames - 1:
                print(f"  FK {t + 1}/{n_frames}")

        flow_arr = np.stack(flow_traj, axis=0)
        np.save(str(FLOW_SAVE_PATH), flow_arr)
        print(f"[LoadJoints] Saved {flow_arr.shape[0]} frames to {FLOW_SAVE_PATH}")

        solved_joints = []
        playback_slider.disabled = True

        if len(flow_traj) >= 2:
            draw_flow_lines(server, flow_traj, name="/flow_lines")
        solve_button.disabled = False

        source.arm_qpos = arm_qpos_arr[-1].astype(np.float64)
        source.gripper_openness = float(grip_arr[-1])
        upload_robot_meshes(server, source, "/source", color=(100, 149, 237))
        source_grip_pc = source.gripper_pc()
        upload_point_cloud(server, "/source_grip_pc", source_grip_pc, color=(0, 200, 255), size=0.004)

    @solve_button.on_click
    def _on_solve(_):
        nonlocal solved_joints
        if not FLOW_SAVE_PATH.exists():
            print("[Solve] No recorded flow found")
            return

        flow_arr = np.load(str(FLOW_SAVE_PATH))  # (T, N_pts, 3)
        n_frames = flow_arr.shape[0]
        print(f"[Solve] Solving {n_frames} frames...")

        # Shift flow to target side
        target_flow = flow_arr - source.origin + target.origin

        # Draw target-side flow lines
        draw_flow_lines(server, list(target_flow), name="/flow_lines_target")

        # Reset target to initial pose
        target.arm_qpos = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float64)
        target.gripper_openness = 1.0

        solved_joints = []
        solver.n_iters = int(iter_slider.value)
        for t in range(n_frames):
            solver.solve(target_flow[t])
            solved_joints.append((target.arm_qpos.copy(), target.gripper_openness))
            if (t + 1) % 10 == 0 or t == n_frames - 1:
                print(f"  Solved {t + 1}/{n_frames}")

        # Enable playback slider
        playback_slider.max = n_frames - 1
        playback_slider.value = 0
        playback_slider.disabled = False

        # Show first frame
        _show_solved_frame(0)
        export_solved_joints_for_robot(solved_joints, ROBOT_TRAJ_SAVE_PATH)
        print(
            f"[Solve] Wrote motor trajectory for hardware: {ROBOT_TRAJ_SAVE_PATH} "
            f"({len(solved_joints)} frames). Play with: python control/flow_solver.py play "
            f"--input {ROBOT_TRAJ_SAVE_PATH.name}"
        )
        print("[Solve] Done. Use the playback slider to scrub.")

    def _show_solved_frame(t_idx: int):
        """Update target robot visualization to solved frame t_idx."""
        if not solved_joints:
            return
        t = int(np.clip(t_idx, 0, len(solved_joints) - 1))
        arm_qpos, openness = solved_joints[t]
        target.arm_qpos = arm_qpos
        target.gripper_openness = openness
        upload_robot_meshes(server, target, "/target", color=(220, 80, 80))
        result_pc = target.gripper_pc()
        upload_point_cloud(server, "/target_grip_pc", result_pc, color=(255, 100, 100), size=0.004)

    @playback_slider.on_update
    def _on_playback(ev):
        _show_solved_frame(int(ev.target.value))

    while True:
        # Read source joint state from sliders
        arm_qpos = np.array([s.value for s in arm_sliders], dtype=np.float64)
        openness = float(grip_slider.value)

        source.arm_qpos = arm_qpos
        source.gripper_openness = openness

        # Check if source changed
        current_key = np.concatenate([arm_qpos, [openness]])
        changed = last_source_qpos is None or not np.allclose(current_key, last_source_qpos, atol=1e-6)

        if changed:
            # Update source visualization
            upload_robot_meshes(server, source, "/source", color=(100, 149, 237))
            source_grip_pc = source.gripper_pc()
            upload_point_cloud(server, "/source_grip_pc", source_grip_pc, color=(0, 200, 255), size=0.004)

            # If recording, append to flow trajectory and draw lines in real time
            if recording:
                flow_traj.append(source_grip_pc.copy())
                if len(flow_traj) >= 2:
                    draw_flow_lines(server, flow_traj, name="/flow_lines")
                print(f"\r[Record] {len(flow_traj)} frames", end="", flush=True)

            last_source_qpos = current_key.copy()

        time.sleep(0.05)


if __name__ == "__main__":
    main()
