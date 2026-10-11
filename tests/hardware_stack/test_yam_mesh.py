"""The export boundary must not silently accept incomplete capture claims."""

import runpy
from pathlib import Path

import pytest

pytest.importorskip("urchin")

pytestmark = pytest.mark.hardware_stack
mesh_target = runpy.run_path(str(Path(__file__).parents[2] / "examples/yam_mesh.py"))["mesh_target"]


@pytest.mark.parametrize("kwargs", [
    {},
    {"side": "left"},
    {"gripper_open": 0.5},
    {"side": "left", "gripper_open": 0.5, "state": {}},
    {"state": {"mode": "unknown", "arms": {}}},
    {"state": {"mode": "synthetic", "arms": {"left": {}}}},
    {"state": {"mode": "live", "arms": {"left": {}, "right": {}}}},
    {"state": {"mode": "live", "arms": {"left": {}, "right": {}},
               "stationary_capture_confirmed": "false"}},
])
def test_invalid_capture_contract_is_rejected_before_loading_model(kwargs):
    with pytest.raises(ValueError):
        mesh_target("must-not-be-loaded.urdf", **kwargs)
