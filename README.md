# lerobot_3d

SO101 multi-camera teleop with live fused point clouds, a viser-based 3D viewer, and calibration tooling.

## Install

```bash
pip install -e ".[realsense]"
```

(`realsense` pulls in `pyrealsense2`, required to stream RealSense cameras.)

## Teleop (`lerobot-teleop`)

Drives **N SO101 follower** arms from **N SO101 leader** teleoperators while streaming **one or more Intel RealSense** cameras, fusing depth into a scene point cloud, sampling the first follower's URDF for a robot point cloud, and rendering all of it live in **viser**.

```bash
lerobot-teleop
```

| Flag | Meaning |
|------|---------|
| `--config PATH` | Teleop config YAML (see below). Omit to use the default `teleop_config.yaml`. |
| `--hz 60` | Main loop rate in Hz (default `60`). Use `0` or negative for no pacing. |

Everything else — recording, extrinsics path, RealSense serials, camera resolution/FPS, the tune panel, the viser port — lives in `teleop_config.yaml`, not on the command line. Run `lerobot-teleop -h` for the full flag list.

Open `http://localhost:<viser_port>` (default `8080`) in a browser to see the fused scene point cloud, the full robot point cloud, and a per-link robot point cloud per URDF link, all updating live. With `tune: true` in the config, the same page shows **Quit** / **Capture** / **Save subgoal** buttons — Quit stops the main loop cleanly, Capture snapshots calibration images (see [Performing calibration](#performing-calibration)), Save subgoal writes the current fused scene to `subgoals/`.

## Teleop configuration

Everything — hardware wiring **and** run settings — lives in **`teleop_config.yaml`**, not in Python. A dev-checkout copy ships at `src/teleop_config.yaml`:

```yaml
leaders:
  - port: /dev/ttyACM0
    id: bender_leader_arm
followers:
  - port: /dev/ttyACM3
    id: bender_follower_arm
realsense_serials:
  - "244622072067"

extrinsic_json: extrinsic_calibration.json
recording_name: ""
tune: true
camera_width: 848
camera_height: 480
camera_fps: 60
viser_port: 8080
```

`leaders`/`followers` must be the same length (matched by list position); `realsense_serials` needs at least one entry. See `src/teleop_config.yaml` for the full, annotated field list (recording, URDF/calibration overrides, camera stream, smoothing) and `lerobot_3d.teleop_config.TeleopSystemConfig` for the underlying dataclass.

**Resolution order** for both `teleop_config.yaml` and the extrinsics JSON: an environment variable (`LEROBOT_3D_TELEOP_CONFIG` / `LEROBOT_3D_EXTRINSIC_JSON`) → the current working directory → `src/<file>` next to the installed package (dev checkout).

## Performing calibration

**Robot arm motor calibration** (homing/joint limits) is handled by LeRobot itself, not this repo — run `lerobot-calibrate` for each leader/follower arm. Point `teleop_config.yaml`'s `robot_calibration_dir` / `robot_calibration_ids` / `robot_calibration_paths` at the resulting JSON if it isn't in LeRobot's default location.

**Camera intrinsics** are written automatically to `intrinsic_calibration.json` when `lerobot-teleop` shuts down and that file doesn't already exist.

**Camera extrinsics** (each RealSense's pose relative to the robot base) are the main calibration workflow:

1. **New camera, no existing entry?** Bootstrap `extrinsic_calibration.json` with an identity transform for its serial:
   ```json
   {
     "YOUR_SERIAL": {
       "X_WC": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
     }
   }
   ```
2. Run `lerobot-teleop`, position the robot arm in view of the camera(s) you're calibrating, and click **Capture** in the viser GUI. This writes `calibration_files/<serial>/{color.png,depth.npz}` per camera and `calibration_files/robot_pcd.npz` (the robot mesh point cloud at that pose).
3. From the same directory (containing `calibration_files/`, `extrinsic_calibration.json`, `intrinsic_calibration.json`), run:
   ```bash
   python -m lerobot_3d.icp
   ```
4. For each camera, an Open3D window opens for manual alignment — **arrows** = XY, **PgUp/PgDn** = Z, **1–6** = rotate about X/Y/Z, **f** = cycle step size, **r** = reset, **Enter** = confirm, **Esc** = abort/revert. ICP then refines the confirmed pose automatically.
5. The refined `extrinsic_calibration.json` is written back — ready for `lerobot-teleop`.

## Custom teleop script

Build a **`TeleopSystemConfig`** (`lerobot_3d.teleop_config`) with **`SO101AxisConfig`** entries for each **leader** and **follower** (`port` + LeRobot `id`), **`realsense_serials`**, and any optional fields you need (`urdf_path`, `robot_calibration_ids`, `camera_width`, `camera_height`, `camera_fps`, `tune`, `viser_port`). `len(leaders)` must equal `len(followers)`. `robot_calibration_ids` defaults to each follower's `id`; the **first** follower's observation drives the mesh/point-cloud visualization returned as `robot_pcd`/`robot_link_pcds`.

Call `step()` each tick for `datapoints` (raw per-camera color/depth), `scene_pcd` (Open3D point cloud — `np.asarray(scene_pcd.points)`/`.colors`), `robot_pcd` (`(M, 3)` `float64`), and `robot_link_pcds` (`dict[str, np.ndarray]` keyed by URDF link name). Call `close()` when `system.viewer.quit` is set:

```python
import time

from lerobot_3d.control.teleop import TeleopPointCloudSystem
from lerobot_3d.teleop_config import load_teleop_system_config

if __name__ == "__main__":
    hz = 15.0
    period_s = None if hz <= 0 else 1.0 / hz

    config = load_teleop_system_config("./my_teleop_config.yaml")

    system = TeleopPointCloudSystem(config)
    system.connect()
    try:
        while not system.viewer.quit:
            t0 = time.monotonic()
            datapoints, scene_pcd, robot_pcd, robot_link_pcds = system.step()
            # use datapoints / scene_pcd / robot_pcd / robot_link_pcds here
            if period_s is not None:
                time.sleep(max(0.0, period_s - (time.monotonic() - t0)))
    finally:
        system.close()
```

`step()` also takes an optional `masks_by_serial` (a `{serial: mask}` dict or a list aligned with `realsense_serials`; nonzero/`True` pixels are kept) to mask the fused point cloud per camera.

For a fully custom stack (different robot type, no `TeleopPointCloudSystem`), build directly on **`SO101Leader`**/**`SO101Follower`** from LeRobot and **`SystemStateViewer`** in `lerobot_3d.point_clouds.system_vis`, passing a `TeleopSystemConfig` and calling `update(*actions)` with one dict per follower each tick.
