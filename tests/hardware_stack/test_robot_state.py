"""Needs open3d/urchin/scipy/lerobot installed. No physical hardware required --
these exercise pure FK/math (via bare-instance construction, bypassing __init__'s
URDF/JSON load) plus one real construction against the bundled URDF/mesh fixtures.
"""
import json

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("urchin")
pytest.importorskip("lerobot")

from lerobot_3d.paths import CALIBRATION_DIR
from lerobot_3d.point_clouds.robot_state import RobotState

pytestmark = pytest.mark.hardware_stack

SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class _FakeURDF:
    """Minimal stand-in for urchin.URDF: only .link_map is touched by the methods under test."""

    def __init__(self, link_names):
        self.link_map = {name: name for name in link_names}


def _bare_robot_state() -> RobotState:
    """A RobotState with none of __init__'s URDF/JSON I/O, for pure-math method tests."""
    return object.__new__(RobotState)


# ---------------------------------------------------------------------------
# ticks_to_radians
# ---------------------------------------------------------------------------


def test_ticks_to_radians_home_pose_is_zero():
    state = _bare_robot_state()
    assert state.ticks_to_radians(2047) == pytest.approx(0.0)


def test_ticks_to_radians_full_revolution_is_two_pi():
    state = _bare_robot_state()
    delta = state.ticks_to_radians(2047 + 4096) - state.ticks_to_radians(2047)
    assert delta == pytest.approx(2 * np.pi)


def test_ticks_to_radians_zero_raw():
    state = _bare_robot_state()
    assert state.ticks_to_radians(0) == pytest.approx(-2047 * (2 * np.pi / 4096))


# ---------------------------------------------------------------------------
# compute_phys_ranges
# ---------------------------------------------------------------------------


def test_compute_phys_ranges(minimal_calibration_dict):
    state = _bare_robot_state()

    ranges = state.compute_phys_ranges(minimal_calibration_dict)

    assert set(ranges) == {"shoulder_pan", "gripper"}
    assert ranges["shoulder_pan"]["drive_mode"] is False
    assert ranges["gripper"]["drive_mode"] is True
    assert ranges["shoulder_pan"]["lo"] == pytest.approx(state.ticks_to_radians(0))
    assert ranges["shoulder_pan"]["hi"] == pytest.approx(state.ticks_to_radians(4095))


def test_compute_phys_ranges_drive_mode_defaults_to_false():
    state = _bare_robot_state()

    ranges = state.compute_phys_ranges({"j": {"range_min": 0, "range_max": 100}})

    assert ranges["j"]["drive_mode"] is False


# ---------------------------------------------------------------------------
# convert_lerobot_action_to_radians
# ---------------------------------------------------------------------------


def test_convert_regular_joint_drive_mode_false_at_max():
    state = _bare_robot_state()
    state.PHYS_RANGES = {"j": {"lo": -1.0, "hi": 1.0, "drive_mode": False}}

    result = state.convert_lerobot_action_to_radians({"j.pos": 100.0})

    assert result["j"] == pytest.approx(1.0)


def test_convert_regular_joint_drive_mode_true_at_max_is_reversed():
    state = _bare_robot_state()
    state.PHYS_RANGES = {"j": {"lo": -1.0, "hi": 1.0, "drive_mode": True}}

    result = state.convert_lerobot_action_to_radians({"j.pos": 100.0})

    assert result["j"] == pytest.approx(-1.0)


def test_convert_regular_joint_clips_above_100():
    state = _bare_robot_state()
    state.PHYS_RANGES = {"j": {"lo": -1.0, "hi": 1.0, "drive_mode": False}}

    result = state.convert_lerobot_action_to_radians({"j.pos": 500.0})

    assert result["j"] == pytest.approx(1.0)


def test_convert_regular_joint_clips_below_negative_100():
    state = _bare_robot_state()
    state.PHYS_RANGES = {"j": {"lo": -1.0, "hi": 1.0, "drive_mode": False}}

    result = state.convert_lerobot_action_to_radians({"j.pos": -500.0})

    assert result["j"] == pytest.approx(-1.0)


def test_convert_gripper_drive_mode_false_at_max():
    state = _bare_robot_state()
    state.PHYS_RANGES = {"gripper": {"lo": 0.0, "hi": 1.0, "drive_mode": False}}

    result = state.convert_lerobot_action_to_radians({"gripper.pos": 100.0})

    assert result["gripper"] == pytest.approx(1.0)


def test_convert_gripper_drive_mode_true_at_max_is_reversed():
    state = _bare_robot_state()
    state.PHYS_RANGES = {"gripper": {"lo": 0.0, "hi": 1.0, "drive_mode": True}}

    result = state.convert_lerobot_action_to_radians({"gripper.pos": 100.0})

    assert result["gripper"] == pytest.approx(0.0)


def test_convert_gripper_clips_negative_to_zero():
    state = _bare_robot_state()
    state.PHYS_RANGES = {"gripper": {"lo": 0.0, "hi": 1.0, "drive_mode": False}}

    result = state.convert_lerobot_action_to_radians({"gripper.pos": -50.0})

    assert result["gripper"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sample_robot_points
# ---------------------------------------------------------------------------


def test_sample_robot_points_transforms_and_stacks():
    state = _bare_robot_state()
    state.robot_urdf = _FakeURDF(["link_a", "link_b"])
    state.link_visual_points = [
        ("link_a", np.array([[0.0, 0.0, 0.0]])),
        ("link_b", np.array([[1.0, 0.0, 0.0]])),
    ]
    translate_x = np.eye(4)
    translate_x[0, 3] = 5.0
    fk_poses = {"link_a": np.eye(4), "link_b": translate_x}

    full, per_link = state.sample_robot_points(fk_poses)

    assert per_link["link_a"] == pytest.approx(np.array([[0.0, 0.0, 0.0]]))
    assert per_link["link_b"] == pytest.approx(np.array([[6.0, 0.0, 0.0]]))
    assert full.shape == (2, 3)


def test_sample_robot_points_empty_returns_zero_shaped_array():
    state = _bare_robot_state()
    state.robot_urdf = _FakeURDF([])
    state.link_visual_points = []

    full, per_link = state.sample_robot_points({})

    assert full.shape == (0, 3)
    assert per_link == {}


# ---------------------------------------------------------------------------
# get_link_poses
# ---------------------------------------------------------------------------


def test_get_link_poses_identity_rotation_translation():
    state = _bare_robot_state()
    state.robot_urdf = _FakeURDF(["link_a"])
    state.link_visual_meshes = [("link_a", np.zeros((1, 3)), np.zeros((0, 3), dtype=int))]

    T = np.eye(4)
    T[:3, 3] = [2.0, 3.0, 4.0]

    poses = state.get_link_poses({"link_a": T})

    translation, quat_wxyz = poses["link_a"]
    assert translation == pytest.approx([2.0, 3.0, 4.0])
    assert quat_wxyz == pytest.approx([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# get_static_meshes
# ---------------------------------------------------------------------------


def test_get_static_meshes_naming_and_enumeration():
    state = _bare_robot_state()
    verts = np.zeros((3, 3))
    faces = np.zeros((1, 3), dtype=int)
    state.link_visual_meshes = [("link_a", verts, faces), ("link_b", verts, faces)]

    meshes = state.get_static_meshes()

    assert [(m[0], m[1]) for m in meshes] == [("link_a", "link_a_0"), ("link_b", "link_b_1")]
    assert np.array_equal(meshes[0][2], verts)
    assert np.array_equal(meshes[0][3], faces)


# ---------------------------------------------------------------------------
# Full construction against the bundled URDF + STL assets (integration-style)
# ---------------------------------------------------------------------------


@pytest.fixture
def so101_calibration_path(tmp_path):
    calib = {name: {"range_min": 0, "range_max": 4095, "drive_mode": 0} for name in SO101_JOINT_NAMES}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calib))
    return path


def test_robot_state_constructs_from_bundled_urdf_and_meshes(so101_calibration_path):
    urdf_path = str(CALIBRATION_DIR / "so101_new_calib.urdf")

    state = RobotState(urdf_path, "so101_test_id", calibration_path=so101_calibration_path)

    assert state.link_visual_points
    assert state.link_visual_meshes

    obs = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
    robot_pcd, robot_link_pcds, link_poses = state.get_robot_state(obs)

    assert robot_pcd.shape[1] == 3
    assert np.isfinite(robot_pcd).all()
    assert robot_link_pcds
    assert link_poses
