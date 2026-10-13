"""Offline YAM URDF: FK, mesh sampling and observation handling; no ROS or motors."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("urchin")
pytest.importorskip("open3d")
pytest.importorskip("viser")

from lerobot_3d.common.types import RobotSnapshot
from lerobot_3d.paths import YAM_URDF_PATH
from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState, joint_keys

pytestmark = pytest.mark.hardware_stack


@pytest.fixture
def station():
    return {side: YamRobotState(YAM_URDF_PATH, side) for side in ("left", "right")}


def _obs(side, q=(0.2, 1.2, 1.5, 0.1, -0.2, 0.3), finger=0.02):
    return {**{f"{side}_joint{i}.pos": v for i, v in enumerate(q, 1)}, f"{side}_gripper.pos": finger}


def test_station_mesh_poses_match_direct_urdf_fk(station):
    for side, state in station.items():
        observation = _obs(side)
        assert list(observation) == joint_keys(side) == state.keys
        joints = state.get_joint_positions(observation)
        assert [joints[f"{side}_joint{i}"] for i in range(1, 7)] == [0.2, 1.2, 1.5, 0.1, -0.2, 0.3]
        assert joints[f"{side}_joint7"] == joints[f"{side}_joint8"] == 0.02  # metres of travel
        snapshot = state.get_robot_snapshot(observation, index=3, base_offset=np.ones(3))
        assert snapshot.index == 3 and snapshot.joint_positions == observation
        base = state.get_link_transform(observation, side + "_base")
        np.testing.assert_allclose(snapshot.base_offset, base[:3, 3])  # station pose, not the grid
        world = state.robot_urdf.link_fk(cfg=joints, use_names=True)
        for name, (translation, quaternion) in snapshot.link_poses.items():
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            local[:3, 3] = translation
            np.testing.assert_allclose(base @ local, world[name], atol=1e-10)
        dense = state.get_mesh_points(observation)
        assert len(dense) > len(snapshot.pcd) and np.isfinite(dense).all()
        moved = state.get_robot_snapshot(_obs(side, q=(0.4, 1.2, 1.5, 0.1, -0.2, 0.3)))
        assert not np.allclose(snapshot.pcd, moved.pcd)
        assert len(state.get_static_meshes()) >= 8


def test_rest_pose_is_all_zero_and_within_limits(station):
    for side, state in station.items():
        rest = state.rest_observation()
        assert rest == dict.fromkeys(joint_keys(side), 0.0)
        assert all(v == 0.0 for v in state.get_joint_positions(rest).values())


def test_noise_at_limits_is_clipped_but_bad_mapping_is_rejected(station):
    state = station["left"]
    # Rest sits on joint 3's lower limit (0); encoder noise just below it is clipped, not rejected.
    joints = state.get_joint_positions({**_obs("left"), "left_joint3.pos": -0.01, "left_gripper.pos": 0.0471})
    assert joints["left_joint3"] == 0.0 and joints["left_joint7"] == pytest.approx(0.04695)
    for bad in ({"left_joint1.pos": 20.0}, {"left_joint2.pos": np.nan}, {"left_gripper.pos": 0.2}):
        with pytest.raises(ValueError):
            state.get_robot_snapshot({**_obs("left"), **bad})
    with pytest.raises(ValueError, match="missing"):
        state.get_joint_positions({"left_joint1.pos": 0.0})


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
