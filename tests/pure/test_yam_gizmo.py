"""YamGizmo against a fake viser server: handles drive the IK targets, and snap to the reached pose."""
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_3d.teleoperators.yam_gizmo import YamGizmo, YamGizmoConfig, _pose, _wxyz


class FakeHandle(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(callbacks={}, removed=False, **kwargs)

    def _register(self, event):
        def decorator(fn):
            self.callbacks[event] = fn
            return fn
        return decorator

    def on_update(self, fn):
        return self._register("update")(fn)

    def on_drag_start(self, fn):
        return self._register("drag_start")(fn)

    def on_drag_end(self, fn):
        return self._register("drag_end")(fn)

    def fire(self, event):
        self.callbacks[event](None)

    def remove(self):
        self.removed = True


class FakeServer:
    def __init__(self):
        self.handles, self.sliders = {}, {}
        self.scene = SimpleNamespace(add_transform_controls=self._handle)
        self.gui = SimpleNamespace(add_slider=self._slider)

    def _handle(self, name, scale, position, wxyz):
        self.handles[name] = FakeHandle(position=np.asarray(position), wxyz=np.asarray(wxyz))
        return self.handles[name]

    def _slider(self, label, min, max, step, initial_value):
        self.sliders[label] = FakeHandle(value=initial_value)
        return self.sliders[label]


@pytest.fixture
def gizmo():
    teleop = YamGizmo(YamGizmoConfig(id="test_gizmo"))
    teleop.connect()
    server = FakeServer()
    teleop.attach(server)
    yield teleop, server
    teleop.disconnect()


def test_pose_round_trip():
    pose = _pose([0.1, -0.2, 0.3], [np.cos(0.3), 0.0, np.sin(0.3), 0.0])
    assert np.allclose(_pose(pose[:3, 3], _wxyz(pose)), pose)


def test_handles_start_at_the_grippers(gizmo):
    teleop, server = gizmo
    for side in teleop.sides:
        handle = server.handles[f"/yam_gizmo/{side}"]
        assert np.allclose(_pose(handle.position, handle.wxyz), teleop._kinematics[side].fk(np.zeros(6)))
    assert set(server.sliders) == {"left gripper", "right gripper"}


def test_dragging_moves_the_arm_and_release_snaps_the_handle(gizmo):
    teleop, server = gizmo
    handle = server.handles["/yam_gizmo/left"]
    kinematics = teleop._kinematics["left"]
    # A reachable target: the gripper with joints 2 and 3 bent a little.
    goal = kinematics.fk(np.array([0.2, 0.4, 0.5, 0.0, 0.0, 0.0]))
    handle.fire("drag_start")
    handle.position, handle.wxyz = goal[:3, 3], _wxyz(goal)
    handle.fire("update")
    for _ in range(5):
        action = teleop.get_action()
    q = np.array([action[f"left_joint{i}.pos"] for i in range(1, 7)])
    assert np.allclose(kinematics.fk(q)[:3, 3], goal[:3, 3], atol=1e-3)
    assert np.allclose(handle.position, goal[:3, 3])          # held where the mouse put it while dragging
    # Out of reach: the handle stays put while dragged, then snaps to the reached pose on release.
    far = goal.copy()
    far[:3, 3] += [2.0, 0.0, 0.0]
    handle.position = far[:3, 3]
    handle.fire("update")
    teleop.get_action()
    assert np.allclose(handle.position, far[:3, 3])
    handle.fire("drag_end")
    teleop.get_action()
    reached = kinematics.fk(teleop._q["left"])
    assert np.allclose(handle.position, reached[:3, 3]) and not np.allclose(reached[:3, 3], far[:3, 3])


def test_slider_sets_the_gripper_and_keys_move_the_handle(gizmo):
    teleop, server = gizmo
    slider = server.sliders["right gripper"]
    slider.value = 0.6
    slider.fire("update")
    assert teleop.get_action()["right_gripper.pos"] == pytest.approx(0.6)
    teleop.active = "right"
    teleop.handle_keys(["G"])                                   # keyboard: open the gripper further
    teleop.get_action()
    assert slider.value == pytest.approx(teleop._gripper["right"]) and slider.value > 0.6


def test_disconnect_removes_handles(gizmo):
    teleop, server = gizmo
    teleop.disconnect()
    assert all(h.removed for h in server.handles.values()) and all(s.removed for s in server.sliders.values())
    teleop.connect()  # the fixture disconnects again
