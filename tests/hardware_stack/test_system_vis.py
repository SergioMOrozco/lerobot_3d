"""Needs open3d/lerobot/viser/pyrealsense2 importable (module-level imports in
system_vis.py and what it pulls in). No physical hardware required -- these exercise
pure interpolation/validation logic via bare-instance construction, bypassing the
hardware-coupled __init__.
"""
import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("lerobot")
pytest.importorskip("viser")
pytest.importorskip("pyrealsense2")

from lerobot_3d.common.types import Datapoint, RobotSnapshot
from lerobot_3d.point_clouds.system_vis import SystemStateViewer, blend_actions, write_intrinsics
from lerobot_3d.point_clouds.viser_viewer import grid_offsets

pytestmark = pytest.mark.hardware_stack


def _bare_viewer() -> SystemStateViewer:
    return object.__new__(SystemStateViewer)


def _datapoint(serial, depth_shape=(2, 2)):
    return Datapoint(
        serial=serial,
        color=None,
        depth=np.zeros(depth_shape),
        depth_scale=1.0,
        max_depth=10.0,
        X_WC=None,
        color_intrinsics=None,
    )


# ---------------------------------------------------------------------------
# _interpolated_actions_locked
# ---------------------------------------------------------------------------


def test_interpolated_actions_locked_returns_none_without_target():
    viewer = _bare_viewer()
    viewer._target_actions = None

    assert viewer._interpolated_actions_locked(now=0.0) is None


def test_interpolated_actions_locked_at_start_time_returns_start():
    viewer = _bare_viewer()
    viewer.action_interpolation_duration_s = 1.0
    viewer._target_start_time = 10.0
    viewer._start_actions = [{"j.pos": 0.0}]
    viewer._target_actions = [{"j.pos": 100.0}]

    result = viewer._interpolated_actions_locked(now=10.0)

    assert result[0]["j.pos"] == pytest.approx(0.0)


def test_interpolated_actions_locked_at_duration_returns_target():
    viewer = _bare_viewer()
    viewer.action_interpolation_duration_s = 1.0
    viewer._target_start_time = 10.0
    viewer._start_actions = [{"j.pos": 0.0}]
    viewer._target_actions = [{"j.pos": 100.0}]

    result = viewer._interpolated_actions_locked(now=11.0)

    assert result[0]["j.pos"] == pytest.approx(100.0)


def test_interpolated_actions_locked_midpoint():
    viewer = _bare_viewer()
    viewer.action_interpolation_duration_s = 2.0
    viewer._target_start_time = 0.0
    viewer._start_actions = [{"j.pos": 0.0}]
    viewer._target_actions = [{"j.pos": 10.0}]

    result = viewer._interpolated_actions_locked(now=1.0)

    assert result[0]["j.pos"] == pytest.approx(5.0)


def test_interpolated_actions_locked_clamped_past_duration():
    viewer = _bare_viewer()
    viewer.action_interpolation_duration_s = 1.0
    viewer._target_start_time = 0.0
    viewer._start_actions = [{"j.pos": 0.0}]
    viewer._target_actions = [{"j.pos": 10.0}]

    result = viewer._interpolated_actions_locked(now=100.0)

    assert result[0]["j.pos"] == pytest.approx(10.0)


def test_interpolated_actions_locked_clamped_before_start():
    viewer = _bare_viewer()
    viewer.action_interpolation_duration_s = 1.0
    viewer._target_start_time = 100.0
    viewer._start_actions = [{"j.pos": 0.0}]
    viewer._target_actions = [{"j.pos": 10.0}]

    result = viewer._interpolated_actions_locked(now=0.0)

    assert result[0]["j.pos"] == pytest.approx(0.0)


def test_interpolated_actions_locked_non_numeric_passthrough():
    viewer = _bare_viewer()
    viewer.action_interpolation_duration_s = 1.0
    viewer._target_start_time = 0.0
    viewer._start_actions = [{"mode": "open"}]
    viewer._target_actions = [{"mode": "closed"}]

    result = viewer._interpolated_actions_locked(now=0.5)

    assert result[0]["mode"] == "closed"


def test_interpolated_actions_locked_sets_current_actions():
    viewer = _bare_viewer()
    viewer.action_interpolation_duration_s = 1.0
    viewer._target_start_time = 0.0
    viewer._start_actions = [{"j.pos": 0.0}]
    viewer._target_actions = [{"j.pos": 10.0}]

    result = viewer._interpolated_actions_locked(now=1.0)

    assert viewer._current_actions == result


# ---------------------------------------------------------------------------
# _apply_masks
# ---------------------------------------------------------------------------


def test_apply_masks_none_is_noop():
    viewer = _bare_viewer()
    datapoints = [_datapoint("s1")]

    viewer._apply_masks(datapoints, None)

    assert datapoints[0].obj_mask is None


def test_apply_masks_mapping():
    viewer = _bare_viewer()
    mask = np.ones((2, 2), dtype=bool)
    datapoints = [_datapoint("s1"), _datapoint("s2")]

    viewer._apply_masks(datapoints, {"s1": mask})

    assert np.array_equal(datapoints[0].obj_mask, mask)
    assert datapoints[1].obj_mask is None


def test_apply_masks_sequence():
    viewer = _bare_viewer()
    mask0 = np.ones((2, 2))
    mask1 = np.zeros((2, 2))
    datapoints = [_datapoint("s1"), _datapoint("s2")]

    viewer._apply_masks(datapoints, [mask0, mask1])

    assert np.array_equal(datapoints[0].obj_mask, mask0)
    assert np.array_equal(datapoints[1].obj_mask, mask1)


def test_apply_masks_sequence_length_mismatch_raises():
    viewer = _bare_viewer()
    datapoints = [_datapoint("s1"), _datapoint("s2")]

    with pytest.raises(ValueError, match="Expected 2 masks"):
        viewer._apply_masks(datapoints, [np.ones((2, 2))])


def test_apply_masks_invalid_type_raises():
    viewer = _bare_viewer()
    datapoints = [_datapoint("s1")]

    with pytest.raises(TypeError):
        viewer._apply_masks(datapoints, 42)


def test_apply_masks_string_input_raises_type_error():
    """Strings are technically Sequences, but must be rejected, not treated as masks."""
    viewer = _bare_viewer()
    datapoints = [_datapoint("s1")]

    with pytest.raises(TypeError):
        viewer._apply_masks(datapoints, "not-a-valid-input")


def test_apply_masks_shape_mismatch_raises():
    viewer = _bare_viewer()
    datapoints = [_datapoint("s1", depth_shape=(4, 4))]

    with pytest.raises(ValueError, match="expected"):
        viewer._apply_masks(datapoints, [np.ones((2, 2))])


# ---------------------------------------------------------------------------
# update() with no followers / no cameras
# ---------------------------------------------------------------------------


class _FakeViewer:
    quit = False
    capture = False
    save_subgoal = False

    def __init__(self):
        self.updates = []

    def update(self, *args):
        self.updates.append(args)


class _FakeRobotState:
    """Stands in for RobotState: every robot gets the same base-frame geometry."""

    def __init__(self):
        self.observations = []

    def get_robot_snapshot(self, obs, index=0, base_offset=None):
        self.observations.append(obs)
        return RobotSnapshot(
            index=index,
            joint_positions=dict(obs),
            joint_radians={},
            pcd=np.zeros((3, 3)),
            link_pcds={"base": np.zeros((1, 3))},
            link_poses={},
            base_offset=np.asarray(base_offset),
        )


def _virtual_robots_viewer(num_robots=1) -> SystemStateViewer:
    viewer = _bare_viewer()
    viewer.viewer = _FakeViewer()
    viewer.robot_states = [_FakeRobotState() for _ in range(num_robots)]
    viewer.num_robots = num_robots
    viewer.base_offsets = grid_offsets(num_robots, spacing=0.5)
    viewer.followers = []
    viewer.observe_only = False
    viewer.robot_type = "so101"
    viewer.stream = None
    viewer.ros_stream = None
    viewer.quit = False
    return viewer


def test_update_without_followers_poses_urdf_from_action():
    viewer = _virtual_robots_viewer()
    action = {"shoulder_pan.pos": 12.0}

    datapoints, scene_pcd, robot_pcds, robot_link_pcds, snapshots = viewer.update(action)

    assert viewer.robot_states[0].observations == [action]
    assert datapoints == []
    assert len(scene_pcd.points) == 0
    assert len(robot_pcds) == 1
    assert robot_pcds[0].shape == (3, 3)
    assert len(viewer.viewer.updates) == 1


def test_update_without_followers_requires_one_action_per_robot():
    viewer = _virtual_robots_viewer(num_robots=1)

    with pytest.raises(ValueError, match="Expected 1 actions"):
        viewer.update({"a.pos": 0.0}, {"a.pos": 1.0})


def test_update_multiple_virtual_robots_returns_one_snapshot_each():
    viewer = _virtual_robots_viewer(num_robots=3)
    actions = [{"a.pos": float(i)} for i in range(3)]

    _, _, robot_pcds, robot_link_pcds, snapshots = viewer.update(*actions)

    assert [s.index for s in snapshots] == [0, 1, 2]
    assert [s.joint_positions for s in snapshots] == actions
    assert [tuple(s.base_offset) for s in snapshots] == [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
    ]
    # Returned clouds are in each robot's base frame -- the grid offset is display-only.
    assert all(np.array_equal(pcd, np.zeros((3, 3))) for pcd in robot_pcds)
    assert len(robot_link_pcds) == 3
    assert viewer.viewer.updates[0][2] == snapshots


def test_update_multiple_virtual_robots_wrong_action_count_raises():
    viewer = _virtual_robots_viewer(num_robots=3)

    with pytest.raises(ValueError, match="Expected 3 actions"):
        viewer.update({"a.pos": 0.0})


# ---------------------------------------------------------------------------
# blend_actions / _blend_from_resume (homing before segmentation)
# ---------------------------------------------------------------------------


def test_blend_actions_midpoint_and_non_numeric():
    result = blend_actions([{"a.pos": 0.0, "mode": "x"}], [{"a.pos": 10.0, "mode": "y"}], 0.5)

    assert result == [{"a.pos": 5.0, "mode": "y"}]


def test_blend_from_resume_passthrough_when_not_homed():
    viewer = _bare_viewer()
    viewer._resume_from = None

    assert viewer._blend_from_resume([{"a.pos": 7.0}]) == [{"a.pos": 7.0}]


def test_blend_from_resume_starts_at_home_then_releases(monkeypatch):
    import lerobot_3d.point_clouds.system_vis as system_vis

    viewer = _bare_viewer()
    viewer._resume_from = ([{"a.pos": 0.0}], None)
    clock = {"t": 100.0}
    monkeypatch.setattr(system_vis.time, "monotonic", lambda: clock["t"])

    assert viewer._blend_from_resume([{"a.pos": 10.0}]) == [{"a.pos": 0.0}]
    clock["t"] += system_vis._RESUME_DURATION_S / 2
    assert viewer._blend_from_resume([{"a.pos": 10.0}]) == [{"a.pos": 5.0}]
    clock["t"] += system_vis._RESUME_DURATION_S
    assert viewer._blend_from_resume([{"a.pos": 10.0}]) == [{"a.pos": 10.0}]
    assert viewer._resume_from is None


# ---------------------------------------------------------------------------
# write_intrinsics
# ---------------------------------------------------------------------------


def test_write_intrinsics_merges_new_serial_into_existing_file(tmp_path):
    import json
    from types import SimpleNamespace

    path = tmp_path / "intrinsic_calibration.json"
    path.write_text(json.dumps({"old": {"fl_x": 1.0}}))
    datapoint = _datapoint("new")
    datapoint.color = np.zeros((4, 6, 3), np.uint8)
    datapoint.color_intrinsics = SimpleNamespace(fx=600.0, fy=601.0, ppx=3.0, ppy=2.0)

    write_intrinsics([datapoint], path=str(path))

    data = json.loads(path.read_text())
    assert data["old"] == {"fl_x": 1.0}
    assert data["new"] == {"fl_x": 600.0, "fl_y": 601.0, "cx": 3.0, "cy": 2.0, "w": 6, "h": 4}
