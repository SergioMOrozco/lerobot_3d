"""YAM keyboard end-effector teleop: kinematics, key handling, and driving the virtual arms."""

import socket

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("urchin")

from lerobot_3d import ros_hub as ros_hub_module
from lerobot_3d.paths import YAM_URDF_PATH
from lerobot_3d.point_clouds.yam_kinematics import YamArmKinematics
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState, joint_keys
from lerobot_3d.teleoperators.yam_keyboard import YamKeyboard, YamKeyboardConfig

pytestmark = pytest.mark.hardware_stack

Q = np.array([0.3, 1.0, 1.2, 0.1, -0.2, 0.4])


@pytest.mark.parametrize("side", ["left", "right"])
def test_fk_matches_urchin_and_ik_reaches_a_nearby_pose(side):
    kinematics = YamArmKinematics(YAM_URDF_PATH, side)
    state = YamRobotState(YAM_URDF_PATH, side)
    obs = {**dict(zip(joint_keys(side), Q)), f"{side}_gripper.pos": 0.0}
    np.testing.assert_allclose(kinematics.fk(Q), state.get_link_transform(obs, f"{side}_gripper"), atol=1e-9)

    target = kinematics.fk(Q)
    target[:3, 3] += [0.02, -0.01, 0.02]
    target[:3, :3] = Rotation.from_euler("z", 10, degrees=True).as_matrix() @ target[:3, :3]
    reached = kinematics.fk(kinematics.ik(target, Q))
    np.testing.assert_allclose(reached, target, atol=1e-5)


@pytest.fixture
def keyboard():
    keyboard = YamKeyboard(YamKeyboardConfig(id="test_keyboard"))
    keyboard.connect()
    return keyboard


def _ee(keyboard, side):
    return keyboard._kinematics[side].fk(np.array([keyboard.get_action()[k] for k in joint_keys(side)[:6]]))


def test_starts_at_rest_and_moves_the_active_end_effector(keyboard):
    action = keyboard.get_action()
    assert list(action) == joint_keys("left") + joint_keys("right")
    assert all(v == 0.0 for v in action.values())
    start = _ee(keyboard, "left")[:3, 3]
    assert keyboard.handle_keys(["r", "r", "w"]) == []
    np.testing.assert_allclose(_ee(keyboard, "left")[:3, 3] - start, [0.005, 0, 0.01], atol=1e-4)
    np.testing.assert_allclose(_ee(keyboard, "right"), keyboard._kinematics["right"].fk(np.zeros(6)))  # untouched
    keyboard.handle_keys(["R"])  # Shift: 4x
    np.testing.assert_allclose(_ee(keyboard, "left")[2, 3] - start[2], 0.01 + 0.02, atol=1e-4)


def test_rotation_keys_turn_about_the_end_effector_axes(keyboard):
    keyboard.handle_keys(["r"] * 6 + ["w"] * 6)  # away from the folded rest pose
    before = _ee(keyboard, "left")
    keyboard.handle_keys(["j"])  # yaw: +3 deg about the end effector's own z
    after = _ee(keyboard, "left")
    np.testing.assert_allclose(after[:3, 3], before[:3, 3], atol=1e-5)
    relative = Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).as_rotvec()
    np.testing.assert_allclose(relative, [0, 0, np.radians(3)], atol=1e-4)


def test_unreachable_targets_do_not_wind_up(keyboard):
    start = _ee(keyboard, "left")[:3, 3]
    for _ in range(10):  # rest is folded against joint limits: it can't go down
        keyboard.handle_keys(["f"])
        keyboard.get_action()
    np.testing.assert_allclose(_ee(keyboard, "left")[:3, 3], start, atol=5e-3)  # closest pose it can reach
    keyboard.handle_keys(["r"])  # so the first press back up moves immediately
    assert _ee(keyboard, "left")[2, 3] > start[2] + 0.004


def test_gripper_tab_and_unused_keys(keyboard):
    assert keyboard.handle_keys(["\n", " ", "y", "n", "g"]) == ["\n", " ", "y", "n"]  # recording keys pass through
    assert keyboard.get_action()["left_gripper.pos"] == pytest.approx(0.05)
    keyboard.handle_keys(["G"] * 10)  # clamps fully open
    assert keyboard.get_action()["left_gripper.pos"] == 1.0
    keyboard.handle_keys(["\t", "g", "r"])
    action = keyboard.get_action()
    assert keyboard.active == "right" and action["right_gripper.pos"] == pytest.approx(0.05)
    assert any(abs(action[k]) > 0 for k in joint_keys("right")[:6])


def test_home_eases_arms_back_to_rest(keyboard, monkeypatch):
    from lerobot_3d.teleoperators import yam_keyboard

    clock = [100.0]
    monkeypatch.setattr(yam_keyboard.time, "monotonic", lambda: clock[0])
    keyboard.handle_keys(["R", "W", "g", "\t", "R", "g"])  # move both arms and open both grippers
    moved = keyboard.get_action()

    keyboard.handle_keys(["0"])  # home the active (right) arm only
    assert keyboard.homing and "homing right" in keyboard.status_text()
    keyboard.handle_keys(["r"])  # motion keys wait while it homes
    clock[0] += 1.0  # halfway through the default 2 s
    halfway = keyboard.get_action()
    for key in joint_keys("right"):
        assert halfway[key] == pytest.approx(moved[key] / 2)
    assert {k: halfway[k] for k in joint_keys("left")} == {k: moved[k] for k in joint_keys("left")}
    clock[0] += 1.5
    done = keyboard.get_action()
    assert all(done[k] == 0.0 for k in joint_keys("right")) and not keyboard.homing

    keyboard.home()  # the Home button: every arm
    clock[0] += 2.0
    assert all(v == 0.0 for v in keyboard.get_action().values())
    keyboard.handle_keys(["r"])  # keys work again once home
    assert keyboard.get_action()["right_joint2.pos"] != 0.0


def test_keyboard_drives_the_virtual_arms_through_the_teleop_system(monkeypatch, tmp_path):
    pytest.importorskip("viser")
    from lerobot_3d.control.teleop import TeleopPointCloudSystem
    from lerobot_3d.teleop_config import TeleopSystemConfig, YamLeaderConfig

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ros_hub_module, "RosHub", lambda: pytest.fail("ROS must not start"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = TeleopSystemConfig(leaders=(YamLeaderConfig(),), followers=(), realsense_serials=(),
                                robot_type="yam", viser_port=port)
    system = TeleopPointCloudSystem(config)
    try:
        system.connect()
        keyboard = system.leaders[0]
        keyboard.handle_keys(["r", "r", "\t", "g"])
        _, _, _, _, snapshots = system.step()
        assert [s.joint_positions for s in snapshots] == system.last_actions
        left, right = system.last_actions
        assert list(left) == joint_keys("left") and list(right) == joint_keys("right")
        assert any(left[k] != 0 for k in joint_keys("left")[:6]) and right["right_gripper.pos"] == pytest.approx(0.05)

        # The viser Home button (clicked on viser's thread) homes every arm on the next step.
        keyboard.config.home_duration_s = 0.0
        system._home_requested = True
        system.step()
        assert all(v == 0.0 for action in system.last_actions for v in action.values())
    finally:
        system.close()


def test_set_joints_starts_from_a_measured_pose(keyboard):
    measured = {**dict(zip(joint_keys("left"), [*Q, 0.6])), "right_joint1.pos": 0.1}  # right is incomplete
    keyboard.set_joints(measured)
    action = keyboard.get_action()
    assert [action[k] for k in joint_keys("left")] == pytest.approx([*Q, 0.6], abs=1e-6)  # holds still
    assert all(action[k] == 0.0 for k in joint_keys("right"))

