"""Export a metric YAM mesh target for the existing camera-calibration files."""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot_3d.point_clouds.yam_robot_state import YamRobotState


def mesh_target(urdf, *, state=None, side=None, gripper_open=None):
    if side is not None:
        if state is not None or gripper_open is None:
            raise ValueError("Gripper export requires an opening and no arm state")
        model = YamRobotState(urdf, side)
        return model.get_mesh_points({"gripper_open": gripper_open}, gripper_only=True)
    if gripper_open is not None or state is None:
        raise ValueError("Full-arm export requires a saved state; gripper export requires --side")
    if state.get("mode") not in ("live", "synthetic"):
        raise ValueError("State mode must be live (saved measurements) or synthetic")
    if set(state.get("arms", {})) != {"left", "right"}:
        raise ValueError("Full-arm export requires both arms")
    if state["mode"] == "live" and state.get("stationary_capture_confirmed") is not True:
        raise ValueError("Confirm the measured joints and images belong to one stationary capture")
    return np.concatenate([
        YamRobotState(urdf, side).get_mesh_points(state["arms"][side])
        for side in ("left", "right")
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True, help="New calibration_files/robot_pcd.npz")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--state", help="Saved paired arm state JSON, for a full-arm target")
    target.add_argument("--side", choices=("left", "right"), help="Gripper-local target")
    parser.add_argument("--gripper-open", type=float, help="0 closed, 1 open; match the visible fingers")
    args = parser.parse_args()
    state = json.loads(Path(args.state).read_text()) if args.state else None
    try:
        points = mesh_target(args.urdf, state=state, side=args.side, gripper_open=args.gripper_open)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, pcd=points)
    frame = args.side + "_gripper" if args.side else "URDF world"
    source = "explicit model opening" if args.side else state["mode"]
    print(f"Saved mesh in metres, frame: {frame}; source: {source}. Not a camera calibration.")


if __name__ == "__main__":
    main()
