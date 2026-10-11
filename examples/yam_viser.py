"""Render a saved bimanual YAM pose; does not connect to hardware."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--state", required=True, help="JSON with mode and both arms")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    metadata = json.loads(Path(args.state).read_text())
    if metadata.get("mode") not in ("live", "synthetic"):
        parser.error("State mode must be live (saved measurements) or synthetic")
    if set(metadata.get("arms", {})) != {"left", "right"}:
        parser.error("Saved state must include both arms")

    models = [YamRobotState(args.urdf, side) for side in ("left", "right")]
    snapshots = [
        model.get_robot_snapshot(metadata["arms"][model.side], index=i)
        for i, model in enumerate(models)
    ]
    viewer = ViserSceneViewer(port=args.port, host="127.0.0.1", controls=False)

    @viewer.server.on_client_connect
    def frame_station(client):
        client.camera.position = (1.6, 0.8, 1.2)
        client.camera.look_at = (0.15, -0.3, 0.35)

    try:
        for model, snapshot in zip(models, snapshots):
            viewer.load_static_meshes(model.get_static_meshes(), snapshot.index, snapshot.base_offset)
        viewer.update(np.empty((0, 3)), None, snapshots)
        viewer.set_status(f"Saved {metadata['mode']} pose; no live feedback or motor commands")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
