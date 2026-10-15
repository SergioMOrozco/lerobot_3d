from pathlib import Path

import pytest

from lerobot_3d.teleop_config import (
    ROS_DEPTH_TOPIC,
    RosCameraConfig,
    SO101AxisConfig,
    TeleopSystemConfig,
    YamArmConfig,
    YamLeaderConfig,
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


def test_validate_axis_sets_zero_leaders_is_valid():
    _validate_axis_sets(
        leaders=(), followers=("f1",), realsense_serials=("s1",), robot_calibration_ids=("c1",)
    )


def test_validate_axis_sets_zero_followers_is_valid():
    _validate_axis_sets(
        leaders=("l1", "l2"), followers=(), realsense_serials=("s1",), robot_calibration_ids=("c1",)
    )


def test_validate_axis_sets_zero_followers_needs_exactly_one_calibration_id():
    with pytest.raises(ValueError, match="robot_calibration_ids must have 1 entry"):
        _validate_axis_sets(
            leaders=(), followers=(), realsense_serials=(), robot_calibration_ids=("c1", "c2")
        )


def test_validate_axis_sets_leader_follower_mismatch():
    with pytest.raises(ValueError, match="must be the same count"):
        _validate_axis_sets(
            leaders=("l1",),
            followers=("f1", "f2"),
            realsense_serials=("s1",),
            robot_calibration_ids=("c1", "c2"),
        )


def test_validate_axis_sets_zero_realsense_serials_is_valid():
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


def test_no_followers_defaults_calibration_to_first_leader():
    config = _minimal_config(followers=())

    assert config.robot_calibration_ids == ("leader_arm",)
    assert config.calibration_robot_type == "so101_leader"


def test_no_followers_explicit_calibration_id_stays_follower_type():
    config = _minimal_config(followers=(), robot_calibration_ids=("custom_id",))

    assert config.robot_calibration_ids == ("custom_id",)
    assert config.calibration_robot_type == "so101_follower"


def test_no_leaders_or_followers_without_calibration_raises():
    with pytest.raises(ValueError, match="set robot_calibration_ids or robot_calibration_paths"):
        _minimal_config(leaders=(), followers=())


def test_no_leaders_or_followers_with_calibration_id_is_valid():
    config = _minimal_config(leaders=(), followers=(), realsense_serials=(), robot_calibration_ids=("x",))

    assert config.robot_calibration_ids == ("x",)


def test_no_leaders_or_followers_with_calibration_path_is_valid():
    config = _minimal_config(leaders=(), followers=(), robot_calibration_paths=["~/calib.json"])

    assert config.robot_calibration_paths == (Path("~/calib.json").expanduser(),)


def _virtual_config(**overrides) -> TeleopSystemConfig:
    kwargs = dict(
        leaders=(),
        followers=(),
        realsense_serials=(),
        robot_calibration_ids=("shared",),
    )
    kwargs.update(overrides)
    return TeleopSystemConfig(**kwargs)


def test_num_robots_defaults_to_one():
    assert _virtual_config().num_robots == 1


def test_num_robots_broadcasts_single_calibration_id():
    config = _virtual_config(num_robots=3)

    assert config.robot_calibration_ids == ("shared", "shared", "shared")


def test_num_robots_broadcasts_single_calibration_path():
    config = _virtual_config(num_robots=2, robot_calibration_paths=["~/calib.json"])

    assert config.robot_calibration_paths == (Path("~/calib.json").expanduser(),) * 2


def test_num_robots_keeps_per_robot_calibration_ids():
    config = _virtual_config(num_robots=2, robot_calibration_ids=("a", "b"))

    assert config.robot_calibration_ids == ("a", "b")


def test_num_robots_wrong_calibration_id_count_raises():
    with pytest.raises(ValueError, match="robot_calibration_ids must have 1 entry"):
        _virtual_config(num_robots=3, robot_calibration_ids=("a", "b"))


@pytest.mark.parametrize(
    "devices",
    [
        {"leaders": (SO101AxisConfig(port="/dev/ttyACM0", id="leader_arm"),)},
        {"followers": (SO101AxisConfig(port="/dev/ttyACM1", id="follower_arm"),)},
    ],
)
def test_num_robots_above_one_with_leader_or_follower_raises(devices):
    with pytest.raises(ValueError, match="num_robots > 1 is only supported"):
        _virtual_config(num_robots=2, **devices)


def test_num_robots_below_one_raises():
    with pytest.raises(ValueError, match="num_robots must be >= 1"):
        _virtual_config(num_robots=0)


def test_non_positive_robot_grid_spacing_raises():
    with pytest.raises(ValueError, match="robot_grid_spacing"):
        _virtual_config(robot_grid_spacing=0.0)


def test_dataset_recording_disabled_by_default():
    config = _minimal_config()

    assert config.dataset_repo_id == ""
    assert config.dataset_fps == 15


def test_non_positive_dataset_fps_raises():
    with pytest.raises(ValueError, match="dataset_fps"):
        _minimal_config(dataset_fps=0)


def test_dataset_recording_with_multiple_robots_raises():
    with pytest.raises(ValueError, match="single robot"):
        _virtual_config(num_robots=2, dataset_repo_id="local/x")


def test_load_teleop_system_config_dataset_fields(teleop_config_yaml_factory):
    path = teleop_config_yaml_factory(
        {"dataset_repo_id": "local/x", "dataset_root": "/tmp/ds", "dataset_task": "push"}
    )

    config = load_teleop_system_config(str(path))

    assert config.dataset_repo_id == "local/x"
    assert config.dataset_root == "/tmp/ds"
    assert config.dataset_task == "push"


def test_segment_on_capture_defaults():
    config = _minimal_config()

    assert config.segment_on_capture is True
    assert config.sam2_model_id == "facebook/sam2.1-hiera-large"


def test_load_teleop_system_config_segment_fields(teleop_config_yaml_factory):
    path = teleop_config_yaml_factory(
        {"segment_on_capture": False, "sam2_model_id": "facebook/sam2.1-hiera-tiny"}
    )

    config = load_teleop_system_config(str(path))

    assert config.segment_on_capture is False
    assert config.sam2_model_id == "facebook/sam2.1-hiera-tiny"


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


def test_load_teleop_system_config_empty_device_lists(teleop_config_yaml_factory):
    path = teleop_config_yaml_factory({"followers": [], "realsense_serials": []})

    config = load_teleop_system_config(str(path))

    assert config.followers == ()
    assert config.realsense_serials == ()
    assert config.robot_calibration_ids == ("leader_arm",)


def test_load_teleop_system_config_ignores_calibration_robot_type_key(teleop_config_yaml_factory):
    path = teleop_config_yaml_factory({"calibration_robot_type": "so101_leader"})

    config = load_teleop_system_config(str(path))

    assert config.calibration_robot_type == "so101_follower"


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


# ---------------------------------------------------------------------------
# robot_type: yam and ros_cameras
# ---------------------------------------------------------------------------

YAM_YAML = {
    "robot_type": "yam",
    "leaders": [],
    "followers": [{"side": "left"}, {"side": "right"}],
    "realsense_serials": [],
    "ros_cameras": [
        {"namespace": "/camera/top"},
        {"namespace": "/camera/left_wrist", "link": "left_camera"},
        {"namespace": "/camera/right_wrist", "link": "right_camera",
         "depth_topic": "aligned_depth_to_color/image_raw/compressedDepth"},
    ],
}


def _yam_config(**overrides) -> TeleopSystemConfig:
    kwargs = dict(leaders=(), followers=(YamArmConfig("left"), YamArmConfig("right")), realsense_serials=(),
                  robot_type="yam")
    kwargs.update(overrides)
    return TeleopSystemConfig(**kwargs)


def test_robot_type_defaults_to_so101():
    config = _minimal_config()
    assert config.robot_type == "so101" and config.ros_cameras == () and config.yam_sides == ()


def test_unknown_robot_type_raises():
    with pytest.raises(ValueError, match="robot_type"):
        _minimal_config(robot_type="ur5")


def test_load_yam_config(teleop_config_yaml_factory):
    config = load_teleop_system_config(teleop_config_yaml_factory(YAM_YAML))
    assert config.followers == (YamArmConfig("left"), YamArmConfig("right"))
    assert config.yam_sides == ("left", "right")
    assert config.followers[0].joint_topic is None  # YamFollowerConfig defaults it to /left/joint_states
    top, left, right = config.ros_cameras
    assert top == RosCameraConfig("/camera/top") and top.link is None and top.depth_topic == ROS_DEPTH_TOPIC
    assert left.link == "left_camera"
    assert right.depth_topic == "aligned_depth_to_color/image_raw/compressedDepth"
    assert config.robot_calibration_ids == ()  # YAM joints are radians; no LeRobot calibration


def test_yam_single_arm(teleop_config_yaml_factory):
    config = load_teleop_system_config(teleop_config_yaml_factory(
        {**YAM_YAML, "followers": [{"side": "right"}], "ros_cameras": YAM_YAML["ros_cameras"][2:]}))
    assert config.yam_sides == ("right",)


def test_yam_without_followers_previews_both_arms():
    assert _yam_config(followers=()).yam_sides == ("left", "right")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"leaders": (SO101AxisConfig("/dev/ttyACM0", "leader"),)}, "one leader"),
        ({"leaders": (YamLeaderConfig(), YamLeaderConfig())}, "one leader"),
        ({"leaders": (YamLeaderConfig("gamepad"),)}, "one leader"),
        ({"followers": (YamArmConfig("left"), YamArmConfig("left"))}, "distinct sides"),
        ({"followers": (YamArmConfig("middle"),)}, "distinct sides"),
        ({"num_robots": 2}, "num_robots"),
        ({"followers": (YamArmConfig("left"),),
          "ros_cameras": (RosCameraConfig("/camera/right_wrist", link="right_camera"),)}, "isn't in followers"),
    ],
)
def test_invalid_yam_configs_raise(overrides, match):
    with pytest.raises(ValueError, match=match):
        _yam_config(**overrides)


def test_linked_ros_camera_needs_yam():
    with pytest.raises(ValueError, match="only supported with robot_type: yam"):
        _minimal_config(ros_cameras=(RosCameraConfig("/camera/wrist", link="gripper_link"),))


def test_fixed_ros_camera_works_with_so101():
    config = _minimal_config(ros_cameras=(RosCameraConfig("/camera/top"),))
    assert config.ros_cameras[0].namespace == "/camera/top"


def test_camera_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        _yam_config(ros_cameras=(RosCameraConfig("/camera/top"), RosCameraConfig("camera/top/")))


def test_ros_camera_yaml_errors(teleop_config_yaml_factory):
    with pytest.raises(ValueError, match="namespace"):
        load_teleop_system_config(teleop_config_yaml_factory({**YAM_YAML, "ros_cameras": [{"link": "left_camera"}]}))
    with pytest.raises(ValueError, match="unknown fields"):
        load_teleop_system_config(teleop_config_yaml_factory(
            {**YAM_YAML, "ros_cameras": [{"namespace": "/camera/top", "serial": "x"}]}))
    with pytest.raises(ValueError, match="side"):
        load_teleop_system_config(teleop_config_yaml_factory({**YAM_YAML, "followers": [{"port": "/dev/x"}]}))


def test_yam_keyboard_leader_drives_virtual_arms(teleop_config_yaml_factory):
    config = load_teleop_system_config(teleop_config_yaml_factory(
        {"robot_type": "yam", "leaders": [{"type": "keyboard"}], "followers": [], "realsense_serials": []}))
    assert config.leaders == (YamLeaderConfig("keyboard"),)
    assert config.yam_sides == ("left", "right")


def test_yam_keyboard_can_drive_real_arms_with_custom_topics(teleop_config_yaml_factory):
    config = load_teleop_system_config(teleop_config_yaml_factory({
        "robot_type": "yam", "leaders": [{"type": "keyboard"}], "realsense_serials": [],
        "followers": [{"side": "left"}, {"side": "right", "joint_topic": "/r/js", "command_topic": "/r/cmd"}],
    }))
    assert config.leaders == (YamLeaderConfig("keyboard"),)
    assert config.followers == (YamArmConfig("left"), YamArmConfig("right", "/r/js", "/r/cmd"))

