from pathlib import Path

import pytest

from lerobot_3d.teleop_config import (
    SO101AxisConfig,
    TeleopSystemConfig,
    _axis_configs,
    _validate_axis_sets,
    load_teleop_system_config,
)

pytestmark = pytest.mark.pure


# ---------------------------------------------------------------------------
# _axis_configs
# ---------------------------------------------------------------------------


def test_axis_configs_happy_path():
    entries = [{"port": "/dev/ttyACM0", "id": "a"}, {"port": "/dev/ttyACM1", "id": "b"}]

    result = _axis_configs(entries, key="leaders")

    assert result == (
        SO101AxisConfig(port="/dev/ttyACM0", id="a"),
        SO101AxisConfig(port="/dev/ttyACM1", id="b"),
    )


def test_axis_configs_empty():
    assert _axis_configs([], key="leaders") == ()


def test_axis_configs_missing_port():
    with pytest.raises(ValueError, match=r"leaders\[0\]"):
        _axis_configs([{"id": "a"}], key="leaders")


def test_axis_configs_missing_id():
    with pytest.raises(ValueError, match=r"followers\[0\]"):
        _axis_configs([{"port": "/dev/ttyACM0"}], key="followers")


# ---------------------------------------------------------------------------
# _validate_axis_sets
# ---------------------------------------------------------------------------


def test_validate_axis_sets_all_valid_raises_nothing():
    _validate_axis_sets(
        leaders=("l1",),
        followers=("f1",),
        realsense_serials=("s1",),
        robot_calibration_ids=("c1",),
        robot_calibration_paths=None,
    )


def test_validate_axis_sets_zero_leaders():
    with pytest.raises(ValueError, match="at least one leader"):
        _validate_axis_sets(
            leaders=(), followers=(), realsense_serials=("s1",), robot_calibration_ids=()
        )


def test_validate_axis_sets_leader_follower_mismatch():
    with pytest.raises(ValueError, match="must be the same count"):
        _validate_axis_sets(
            leaders=("l1",),
            followers=("f1", "f2"),
            realsense_serials=("s1",),
            robot_calibration_ids=("c1", "c2"),
        )


def test_validate_axis_sets_zero_realsense_serials():
    with pytest.raises(ValueError, match="at least one RealSense serial"):
        _validate_axis_sets(
            leaders=("l1",), followers=("f1",), realsense_serials=(), robot_calibration_ids=("c1",)
        )


def test_validate_axis_sets_robot_calibration_ids_mismatch():
    with pytest.raises(ValueError, match="robot_calibration_ids must have one entry"):
        _validate_axis_sets(
            leaders=("l1",),
            followers=("f1",),
            realsense_serials=("s1",),
            robot_calibration_ids=("c1", "c2"),
        )


def test_validate_axis_sets_robot_calibration_paths_mismatch():
    with pytest.raises(ValueError, match="robot_calibration_paths must have one entry"):
        _validate_axis_sets(
            leaders=("l1",),
            followers=("f1",),
            realsense_serials=("s1",),
            robot_calibration_ids=("c1",),
            robot_calibration_paths=("p1", "p2"),
        )


# ---------------------------------------------------------------------------
# TeleopSystemConfig.__post_init__
# ---------------------------------------------------------------------------


def _minimal_config(**overrides) -> TeleopSystemConfig:
    kwargs = dict(
        leaders=(SO101AxisConfig(port="/dev/ttyACM0", id="leader_arm"),),
        followers=(SO101AxisConfig(port="/dev/ttyACM1", id="follower_arm"),),
        realsense_serials=("000000000000",),
    )
    kwargs.update(overrides)
    return TeleopSystemConfig(**kwargs)


def test_minimal_config_defaults():
    config = _minimal_config()

    assert config.robot_calibration_ids == ("follower_arm",)
    assert config.extrinsic_json == "extrinsic_calibration.json"
    assert config.tune is True


@pytest.mark.parametrize("field", ["camera_width", "camera_height", "camera_fps"])
def test_non_positive_camera_fields_raise(field):
    with pytest.raises(ValueError, match="must be positive"):
        _minimal_config(**{field: 0})


def test_negative_action_interpolation_duration_raises():
    with pytest.raises(ValueError, match="action_interpolation_duration_s"):
        _minimal_config(action_interpolation_duration_s=-0.1)


def test_non_positive_action_command_hz_raises():
    with pytest.raises(ValueError, match="action_command_hz"):
        _minimal_config(action_command_hz=0)


def test_non_positive_viser_port_raises():
    with pytest.raises(ValueError, match="viser_port"):
        _minimal_config(viser_port=0)


def test_explicit_robot_calibration_ids_preserved():
    config = _minimal_config(robot_calibration_ids=["custom_id"])

    assert config.robot_calibration_ids == ("custom_id",)


def test_robot_calibration_paths_get_expanded():
    config = _minimal_config(robot_calibration_paths=["~/calib.json"])

    assert config.robot_calibration_paths == (Path("~/calib.json").expanduser(),)


def test_lists_get_coerced_to_tuples():
    config = TeleopSystemConfig(
        leaders=[SO101AxisConfig(port="/dev/ttyACM0", id="a")],
        followers=[SO101AxisConfig(port="/dev/ttyACM1", id="b")],
        realsense_serials=["000000000000"],
    )

    assert isinstance(config.leaders, tuple)
    assert isinstance(config.followers, tuple)
    assert isinstance(config.realsense_serials, tuple)


# ---------------------------------------------------------------------------
# load_teleop_system_config
# ---------------------------------------------------------------------------


def test_load_teleop_system_config_end_to_end(teleop_config_yaml_factory):
    path = teleop_config_yaml_factory({"camera_width": 640, "tune": False})

    config = load_teleop_system_config(str(path))

    assert config.leaders == (SO101AxisConfig(port="/dev/ttyACM0", id="leader_arm"),)
    assert config.followers == (SO101AxisConfig(port="/dev/ttyACM1", id="follower_arm"),)
    assert config.realsense_serials == ("000000000000",)
    assert config.camera_width == 640
    assert config.tune is False


def test_load_teleop_system_config_ignores_unknown_keys(teleop_config_yaml_factory):
    path = teleop_config_yaml_factory({"totally_unknown_field": "x"})

    config = load_teleop_system_config(str(path))

    assert not hasattr(config, "totally_unknown_field")


def test_load_teleop_system_config_none_values_dont_override_defaults(teleop_config_yaml_factory):
    path = teleop_config_yaml_factory({"tune": None})

    config = load_teleop_system_config(str(path))

    assert config.tune is True


def test_load_teleop_system_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_teleop_system_config(str(tmp_path / "does_not_exist.yaml"))
