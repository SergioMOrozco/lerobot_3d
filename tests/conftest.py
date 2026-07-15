import pytest
import yaml


@pytest.fixture(autouse=True)
def _clear_lerobot3d_env(monkeypatch):
    """Prevent a developer's real shell env from leaking into path-resolution tests."""
    monkeypatch.delenv("LEROBOT_3D_EXTRINSIC_JSON", raising=False)
    monkeypatch.delenv("LEROBOT_3D_TELEOP_CONFIG", raising=False)


@pytest.fixture
def minimal_calibration_dict():
    """A tiny but valid SO101 calibration dict: a couple of joints with range_min/max."""
    return {
        "shoulder_pan": {"range_min": 0, "range_max": 4095, "drive_mode": 0},
        "gripper": {"range_min": 1024, "range_max": 3072, "drive_mode": 1},
    }


@pytest.fixture
def teleop_config_yaml_factory(tmp_path):
    """Write a teleop_config.yaml-shaped file from an overrides dict; return its path."""

    def _write(overrides: dict, filename: str = "teleop_config.yaml"):
        base = {
            "leaders": [{"port": "/dev/ttyACM0", "id": "leader_arm"}],
            "followers": [{"port": "/dev/ttyACM1", "id": "follower_arm"}],
            "realsense_serials": ["000000000000"],
        }
        base.update(overrides)
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(base))
        return path

    return _write
