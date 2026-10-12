"""Offline YAM URDF, Viser and mesh calibration checks; no motor interface."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("urchin")
pytest.importorskip("open3d")
pytest.importorskip("viser")

from lerobot_3d.common.types import RobotSnapshot
from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState

pytestmark = pytest.mark.hardware_stack


@pytest.fixture
def station():
    checkout = os.environ.get("I2RT_CHECKOUT")
    if not checkout:
        pytest.skip("Set I2RT_CHECKOUT to the pinned vendor checkout for real URDF tests")
    path = (
        Path(checkout)
        / "i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf"
    )
    return {side: YamRobotState(path, side) for side in ("left", "right")}


@pytest.fixture
def observation():
    return {"position_rad": [0.2, 1.2, 1.5, 0.1, -0.2, 0.3], "gripper_open": 0.4}


def test_station_mesh_poses_match_direct_urdf_fk(station, observation):
    for side, state in station.items():
        joints = state.get_joint_positions(observation)
        assert [joints[f"{side}_joint{i}"] for i in range(1, 7)] == observation["position_rad"]
        assert joints[f"{side}_joint7"] == pytest.approx(0.4 * 0.04695)
        assert joints[f"{side}_joint8"] == pytest.approx(0.4 * 0.04695)
        snapshot = state.get_robot_snapshot(observation)
        base = state.get_link_transform(observation, side + "_base")
        world = state.robot_urdf.link_fk(cfg=joints, use_names=True)
        for name, (translation, quaternion) in snapshot.link_poses.items():
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            local[:3, 3] = translation
            np.testing.assert_allclose(base @ local, world[name], atol=1e-10)
        dense = state.get_mesh_points(observation)
        assert len(dense) > len(snapshot.pcd) and np.isfinite(dense).all()
        changed = {**observation, "position_rad": [0.4, *observation["position_rad"][1:]]}
        moved = state.get_robot_snapshot(changed)
        assert not np.allclose(snapshot.pcd, moved.pcd)
        assert len(state.get_static_meshes()) >= 8


def test_bad_feedback_is_rejected(station, observation):
    state = station["left"]
    for bad in ([0] * 5, [np.nan] * 6, [20] * 6):
        with pytest.raises(ValueError):
            state.get_robot_snapshot({**observation, "position_rad": bad})
    with pytest.raises(ValueError):
        state.get_robot_snapshot({**observation, "gripper_open": 1.1})


def test_viser_applies_base_rotation_and_so101_default_is_identity():
    viewer = ViserSceneViewer.__new__(ViserSceneViewer)
    frame = SimpleNamespace(position=None, wxyz=None)
    viewer._scene_handle = None
    viewer._robot_frames = {0: frame}
    viewer._robot_handles = {}
    viewer._upsert_point_cloud = lambda *args: None
    viewer.update_link_poses = lambda *args, **kwargs: None
    snapshot = RobotSnapshot(0, {}, {}, np.zeros((1, 3)), {}, {}, np.array([1.0, 2.0, 3.0]))
    assert snapshot.base_wxyz == (1, 0, 0, 0)
    snapshot.base_wxyz = tuple(Rotation.from_euler("z", 0.7).as_quat()[[3, 0, 1, 2]])
    viewer.update(np.empty((0, 3)), None, [snapshot])
    np.testing.assert_array_equal(frame.position, snapshot.base_offset)
    np.testing.assert_array_equal(frame.wxyz, snapshot.base_wxyz)


def test_gripper_target_uses_only_local_gripper_geometry_without_arm_feedback(station):
    for side, state in station.items():
        local = state.get_mesh_points({"gripper_open": 0.4}, gripper_only=True)
        assert local.shape == (30000, 3)
        assert np.isfinite(local).all()
        # Independently compose the selected cached visual surfaces at a nonzero
        # arm pose; the flange-relative target must be identical.
        observation = {"position_rad": [0.4, 1.3, 1.1, 0.2, -0.3, 0.1], "gripper_open": 0.4}
        poses = state.robot_urdf.link_fk(cfg=state.get_joint_positions(observation), use_names=True)
        inv = np.linalg.inv(poses[side + "_gripper"])
        expected = []
        for name, points in state._calibration_points:
            if name in {side + s for s in ("_gripper", "_tip_left", "_tip_right")}:
                matrix = inv @ poses[name]
                expected.append(np.einsum("ij,nj->ni", matrix[:3, :3], points) + matrix[:3, 3])
        np.testing.assert_allclose(local, np.concatenate(expected), atol=1e-10)
        opened = state.get_mesh_points({"gripper_open": 0.9}, gripper_only=True)
        np.testing.assert_allclose(local[:10000], opened[:10000])  # fixed housing
        assert not np.allclose(local[10000:], opened[10000:])  # articulated fingers
        with pytest.raises(ValueError):
            state.get_mesh_points({"gripper_open": float("nan")}, gripper_only=True)
