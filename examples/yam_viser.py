"""Preview a saved YAM pose or export its mesh for the existing ICP workflow."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState


def load(path):
    return json.loads(Path(path).read_text())


def frame_station(server):
    @server.on_client_connect
    def connected(client):
        client.camera.position = (1.6, 0.8, 1.2)
        client.camera.look_at = (0.15, -0.3, 0.35)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("view", "export-mesh", "export-gripper"))
    parser.add_argument("--urdf", required=True, help="I2RT composed YAM v1 / linear_4310 station URDF")
    parser.add_argument("--state", help="Saved paired state JSON; required for full-arm modes")
    parser.add_argument("--side", choices=("left", "right"), help="Gripper to export")
    parser.add_argument(
        "--gripper-open", type=float, help="URDF opening 0=closed, 1=open; match the captured fingers"
    )
    parser.add_argument("--output", help="New robot_pcd.npz for the existing icp.py workflow")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.mode == "export-gripper":
        if args.side is None or args.gripper_open is None or not args.output:
            parser.error("export-gripper requires --side, --gripper-open and --output")
        state = YamRobotState(args.urdf, args.side)
        points = state.get_mesh_points({"gripper_open": args.gripper_open}, gripper_only=True)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            np.savez_compressed(stream, pcd=points)
        print(
            f"Saved {args.side} gripper-local target in metres; opening is model input, not joint feedback."
        )
        return
    if not args.state:
        parser.error("Full-arm modes require --state")
    metadata = load(args.state)
    if metadata.get("mode") not in ("live", "synthetic"):
        parser.error("State mode must be live (saved measurements) or synthetic (offline example)")
    observations = metadata["arms"]
    if set(observations) != {"left", "right"}:
        parser.error("Saved state must include both arms")
    states = {side: YamRobotState(args.urdf, side) for side in observations}
    snapshots = [
        state.get_robot_snapshot(observations[side], index=i)
        for i, (side, state) in enumerate(states.items())
    ]
    if args.mode == "view":
        viewer = ViserSceneViewer(port=args.port, host="127.0.0.1", controls=False)
        frame_station(viewer.server)
        try:
            for state, snapshot in zip(states.values(), snapshots):
                viewer.load_static_meshes(state.get_static_meshes(), snapshot.index, snapshot.base_offset)
            viewer.update(np.empty((0, 3)), None, snapshots)
            viewer.set_status(f"Saved {metadata['mode']} snapshot; no live feedback or motor commands")
            print("Ctrl-C to close the saved-pose preview.")
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()
        return
    if not args.output:
        parser.error("export-mesh requires --output")
    if metadata["mode"] == "live" and not metadata.get("stationary_capture_confirmed"):
        parser.error("Confirm the saved joints and camera images belong to one stationary capture")
    points = np.concatenate([state.get_mesh_points(observations[side]) for side, state in states.items()])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, pcd=points)
    print(f"Saved {metadata['mode']} YAM mesh to {output}; use the existing icp.py calibration workflow.")


if __name__ == "__main__":
    main()
