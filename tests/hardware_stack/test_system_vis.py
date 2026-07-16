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

from lerobot_3d.common.types import Datapoint
from lerobot_3d.point_clouds.system_vis import SystemStateViewer

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
