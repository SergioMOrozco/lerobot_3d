"""Shared USB / device identifiers for SO101 teleop stacks (no I/O imports)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class SO101AxisConfig:
    """One SO101 arm on a serial device (leader teleop or follower robot)."""

    port: str
    """Device path, e.g. ``/dev/ttyACM0``."""
    id: str
    """LeRobot calibration / bus id, e.g. ``bender_leader_arm``."""


DEFAULT_SO101_LEADERS: tuple[SO101AxisConfig, ...] = (
    SO101AxisConfig("/dev/ttyACM0", "bender_leader_arm"),
    SO101AxisConfig("/dev/ttyACM1", "clamps_leader_arm"),
)
DEFAULT_SO101_FOLLOWERS: tuple[SO101AxisConfig, ...] = (
    SO101AxisConfig("/dev/ttyACM3", "bender_follower_arm"),
    SO101AxisConfig("/dev/ttyACM2", "clamps_follower_arm"),
)
DEFAULT_REALSENSE_SERIALS: tuple[str, ...] = ("244622072067", "044322073544")


def _validate_axis_sets(
    leaders: Sequence[SO101AxisConfig],
    followers: Sequence[SO101AxisConfig],
    realsense_serials: Sequence[str],
    robot_calibration_ids: Sequence[str],
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


@dataclass(frozen=True)
class TeleopSystemConfig:
    """Everything needed to construct :class:`TeleopPointCloudSystem` / :class:`SystemStateViewer`."""

    realsense_serials: tuple[str, ...] = DEFAULT_REALSENSE_SERIALS
    extrinsic_json: str = "extrinsic_calibration.json"
    leaders: tuple[SO101AxisConfig, ...] = DEFAULT_SO101_LEADERS
    followers: tuple[SO101AxisConfig, ...] = DEFAULT_SO101_FOLLOWERS
    recording_name: str = ""
    """Non-empty → write ``recordings/<name>/`` on shutdown."""
    urdf_path: str | None = None
    """``None`` → bundled ``so101_new_calib.urdf`` under package ``calibration/``."""
    robot_calibration_ids: tuple[str, ...] | None = None
    """HF / LeRobot calibration name per follower; ``None`` → each follower's ``id``."""
    tune: bool = True
    point_size: float = 2.0
    publish_to_foxglove: bool = True
    """If true, start Foxglove and publish point clouds / transforms."""
    display_point_cloud_viewer: bool = False
    """If true, show full scene + robot clouds in the Open3D point cloud viewer."""
    mask_provider: Callable[[list[dict[str, Any]]], Mapping[str, Any] | Sequence[Any] | None] | None = None
    """Optional callback that returns one mask per camera datapoint before fusion."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "realsense_serials", tuple(self.realsense_serials))
        object.__setattr__(self, "leaders", tuple(self.leaders))
        object.__setattr__(self, "followers", tuple(self.followers))
        rc = self.robot_calibration_ids
        if rc is None:
            rc = tuple(f.id for f in self.followers)
            object.__setattr__(self, "robot_calibration_ids", rc)
        else:
            object.__setattr__(self, "robot_calibration_ids", tuple(rc))
        _validate_axis_sets(
            self.leaders,
            self.followers,
            self.realsense_serials,
            self.robot_calibration_ids,
        )
