"""``TeleopSystemConfig``: everything needed to run the teleop + point cloud system.

Values live in ``teleop_config.yaml``, not here -- see :func:`load_teleop_system_config`.
Edit that file (or point ``--config`` / ``LEROBOT_PLAYGROUND_TELEOP_CONFIG`` at your own copy)
to change hardware wiring or run settings; no code changes needed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from lerobot_playground.paths import resolve_teleop_config_yaml

DEFAULT_TELEOP_CONFIG_YAML = "teleop_config.yaml"
"""Default filename for the editable system config; resolved via
:func:`lerobot_playground.paths.resolve_teleop_config_yaml`."""


@dataclass(frozen=True)
class SO101AxisConfig:
    """One SO101 arm on a serial device (leader teleop or follower robot)."""

    port: str
    """Device path, e.g. ``/dev/ttyACM0``."""
    id: str
    """LeRobot calibration / bus id, e.g. ``bender_leader_arm``."""


def _axis_configs(entries: Sequence[Mapping[str, Any]], *, key: str) -> tuple[SO101AxisConfig, ...]:
    axes = []
    for i, entry in enumerate(entries):
        try:
            axes.append(SO101AxisConfig(port=str(entry["port"]), id=str(entry["id"])))
        except KeyError as e:
            raise ValueError(f"{DEFAULT_TELEOP_CONFIG_YAML}: {key}[{i}] missing required field {e}") from e
    return tuple(axes)


def _validate_axis_sets(
    leaders: Sequence[SO101AxisConfig],
    followers: Sequence[SO101AxisConfig],
    realsense_serials: Sequence[str],
    robot_calibration_ids: Sequence[str],
    robot_calibration_paths: Sequence[str | Path] | None = None,
) -> None:
    if len(leaders) < 1:
        raise ValueError("Need at least one leader (teleop).")
    if len(leaders) != len(followers):
        raise ValueError(
            f"Leaders ({len(leaders)}) and followers ({len(followers)}) must be the same count."
        )
    if len(realsense_serials) < 1:
        raise ValueError("Need at least one RealSense serial.")
    if len(robot_calibration_ids) != len(followers):
        raise ValueError(
            "robot_calibration_ids must have one entry per follower "
            f"(got {len(robot_calibration_ids)} ids for {len(followers)} followers)."
        )
    if robot_calibration_paths is not None and len(robot_calibration_paths) != len(followers):
        raise ValueError(
            "robot_calibration_paths must have one entry per follower "
            f"(got {len(robot_calibration_paths)} paths for {len(followers)} followers)."
        )


@dataclass(frozen=True)
class TeleopSystemConfig:
    """Everything needed to construct :class:`TeleopPointCloudSystem` / :class:`SystemStateViewer`.

    Construct this via :func:`load_teleop_system_config` in normal use; build it directly only
    for scripts/tests that don't want to touch a YAML file.
    """

    leaders: tuple[SO101AxisConfig, ...]
    followers: tuple[SO101AxisConfig, ...]
    realsense_serials: tuple[str, ...]
    extrinsic_json: str = "extrinsic_calibration.json"
    recording_name: str = ""
    """Non-empty → write ``recordings/<name>/`` on shutdown."""
    urdf_path: str | None = None
    """``None`` → bundled ``so101_new_calib.urdf`` under package ``calibration/``."""
    robot_calibration_ids: tuple[str, ...] | None = None
    """HF / LeRobot calibration name per follower; ``None`` → each follower's ``id``."""
    robot_calibration_dir: str | Path | None = None
    """Directory containing ``<robot_calibration_id>.json``; ``None`` uses LeRobot defaults."""
    robot_calibration_paths: tuple[str | Path, ...] | None = None
    """Explicit calibration JSON path per follower; overrides ``robot_calibration_dir``."""
    tune: bool = True
    """If true, show the viser GUI's Quit/Capture/Save-subgoal controls."""
    point_size: float = 0.003
    """Viser point radius in world-space meters (not pixels)."""
    camera_width: int = 848
    camera_height: int = 480
    camera_fps: int = 60
    action_interpolation_duration_s: float = 0.12
    """Seconds to blend from the current command to a new target. ``0`` disables smoothing."""
    action_command_hz: float = 50.0
    """Follower command loop rate when action interpolation is enabled."""
    viser_port: int = 8080
    """Port for the viser server hosting the scene + robot point clouds and GUI controls."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "realsense_serials", tuple(self.realsense_serials))
        object.__setattr__(self, "leaders", tuple(self.leaders))
        object.__setattr__(self, "followers", tuple(self.followers))
        if self.camera_width <= 0 or self.camera_height <= 0 or self.camera_fps <= 0:
            raise ValueError("camera_width, camera_height, and camera_fps must be positive.")
        if self.action_interpolation_duration_s < 0:
            raise ValueError("action_interpolation_duration_s must be >= 0.")
        if self.action_command_hz <= 0:
            raise ValueError("action_command_hz must be positive.")
        if self.viser_port <= 0:
            raise ValueError("viser_port must be positive.")
        if self.robot_calibration_ids is None:
            object.__setattr__(self, "robot_calibration_ids", tuple(f.id for f in self.followers))
        else:
            object.__setattr__(self, "robot_calibration_ids", tuple(self.robot_calibration_ids))
        if self.robot_calibration_paths is not None:
            object.__setattr__(
                self,
                "robot_calibration_paths",
                tuple(Path(p).expanduser() for p in self.robot_calibration_paths),
            )
        _validate_axis_sets(
            self.leaders,
            self.followers,
            self.realsense_serials,
            self.robot_calibration_ids,
            self.robot_calibration_paths,
        )


_SPECIAL_FIELDS = {"leaders", "followers", "realsense_serials"}


def load_teleop_system_config(path: str | Path | None = None) -> TeleopSystemConfig:
    """Load a full :class:`TeleopSystemConfig` from ``teleop_config.yaml``.

    Uses the same search order as camera extrinsics (env var, cwd, package-adjacent ``src/``);
    see :func:`lerobot_playground.paths.resolve_teleop_config_yaml`. Pass ``None`` (or omit) to
    use the default filename, :data:`DEFAULT_TELEOP_CONFIG_YAML`.
    """
    resolved = resolve_teleop_config_yaml(DEFAULT_TELEOP_CONFIG_YAML if path is None else path)
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}

    known_fields = {f.name for f in dataclasses.fields(TeleopSystemConfig)} - _SPECIAL_FIELDS
    kwargs = {k: v for k, v in data.items() if k in known_fields and v is not None}

    return TeleopSystemConfig(
        leaders=_axis_configs(data.get("leaders") or [], key="leaders"),
        followers=_axis_configs(data.get("followers") or [], key="followers"),
        realsense_serials=tuple(str(s) for s in data.get("realsense_serials") or []),
        **kwargs,
    )
