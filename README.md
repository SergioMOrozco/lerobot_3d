# lerobot_3d

SO101 multi-camera teleop, fused point clouds, a viser-based live viewer, and related calibration utilities.

## Install

**Editable** (recommended while developing):

```bash
cd /path/to/lerobot_3d
pip install -e .
```

**Into another environment** from a checkout or wheel:

```bash
pip install /path/to/lerobot_3d
```

**RealSense** cameras are required for teleop; install the extra so `pyrealsense2` is available:

```bash
pip install -e ".[realsense]"
```

Import the package in Python as `lerobot_3d`, for example:

```python
from lerobot_3d.paths import CALIBRATION_DIR
```

## Teleop (`lerobot-teleop`)

Teleop drives **N SO101 follower** arms from **N SO101 leader** teleoperators while streaming **one or more Intel RealSense** cameras, fusing depth into a scene point cloud, sampling the first follower’s URDF for a robot point cloud, and rendering all of it live in **viser** (a browser-based viewer server, started automatically).

### What you need

- `pyrealsense2` (use `pip install -e ".[realsense]"`).
- **Camera extrinsics** in JSON (see below). Intrinsics can be written on shutdown when missing (see `SystemStateViewer.close`).
- A **`teleop_config.yaml`** (see below) describing the full **`TeleopSystemConfig`**: RealSense serials, leader/follower `port` + `id`, and run settings (recording, camera stream, viser port, smoothing).

### Teleop config

Everything needed to run teleop — hardware wiring **and** run settings — lives in **`teleop_config.yaml`**, not in Python. Edit that file when your USB layout or run settings change; no code changes needed. Same resolution order as the extrinsics JSON (below), with its own env var **`LEROBOT_3D_TELEOP_CONFIG`**. A dev-checkout copy ships at **`src/teleop_config.yaml`**:

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

`leaders`/`followers` must be the same length (matched by list position); `realsense_serials` needs at least one entry. See `src/teleop_config.yaml` for the full set of fields and their defaults, or `lerobot_3d.teleop_config.TeleopSystemConfig` for field docs.

### Calibration files

Extrinsics are **not** bundled in the package. Resolution order for the default filename `extrinsic_calibration.json`:

1. Environment variable **`LEROBOT_3D_EXTRINSIC_JSON`** (absolute path to the file).
2. File in the **current working directory**.
3. In a dev checkout, **`src/extrinsic_calibration.json`** next to the installed package tree.

Set an explicit path via the `extrinsic_json` field in `teleop_config.yaml`.

### How to run

```bash
lerobot-teleop
```

Flags:

| Flag | Meaning |
|------|---------|
| `--config PATH` | Teleop config YAML (see above). Omit to use the default `teleop_config.yaml`. |
| `--hz 60` | Main loop rate in Hz (default `60`). Use `0` or negative for no pacing. This is a run-loop setting, not part of the config file. |

Everything else — recording, extrinsics path, RealSense serials, camera resolution/FPS, the tune panel, and the viser port — is set in `teleop_config.yaml`, not on the command line.

Examples:

```bash
# Default config file, default rate
lerobot-teleop

# Use a different config file
lerobot-teleop --config ./my_teleop_config.yaml

# Slower loop
lerobot-teleop --hz 10
```

Open `http://localhost:<viser_port>` (default `8080`) in a browser to view the fused scene point cloud, the full robot point cloud, and a per-link robot point cloud for each URDF link, all updating live. With `tune: true` in the config, the same page also shows Quit / Capture / Save-subgoal buttons — Quit stops the main loop cleanly, Capture snapshots calibration images, and Save-subgoal writes the current fused scene to `subgoals/`.

### Custom teleop script

Build a **`TeleopSystemConfig`** (`lerobot_3d.teleop_config`) with **`SO101AxisConfig`** entries for each **leader** and **follower** (`port` + LeRobot `id`), **`realsense_serials`**, and optional fields (`urdf_path`, `robot_calibration_ids`, `camera_width`, `camera_height`, `camera_fps`, `tune`, `viser_port`). **`len(leaders)` must equal `len(followers)`** (one leader action per follower). **`robot_calibration_ids`** defaults to each follower’s `id`; the **first** follower’s observation drives the mesh / point-cloud visualization returned as **`robot_pcd`** and **`robot_link_pcds`**. Robot mesh points are sampled once at startup and then transformed by FK every step.

**Minimal** (same behavior as the CLI defaults, but from your own file): call **`step()`** each tick for **`datapoints`**, **`scene_pcd`**, **`robot_pcd`**, and **`robot_link_pcds`**. `datapoints` contains the raw per-camera color/depth data used for fusion. `scene_pcd` is the full Open3D point cloud, so you can read both `np.asarray(scene_pcd.points)` and `np.asarray(scene_pcd.colors)`. `robot_pcd` is an **`(M, 3)`** **`float64`** array, and `robot_link_pcds` is a **`dict[str, np.ndarray]`** keyed by URDF link name. You can pass optional image masks into `step(masks_by_serial=...)`; nonzero / `True` pixels are kept. Call **`close()`** when `viewer.quit` is set.

Load from a YAML file with `load_teleop_system_config` (same loader `lerobot-teleop --config` uses):

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
            scene_points = np.asarray(scene_pcd.points)
            scene_colors = np.asarray(scene_pcd.colors)
            # use datapoints / scene_points / scene_colors / robot_pcd / robot_link_pcds here
            if period_s is not None:
                time.sleep(max(0.0, period_s - (time.monotonic() - t0)))
    finally:
        system.close()
```

If you already have masks for the current images, pass either a serial-keyed dict or a list aligned with `realsense_serials`:

```python
masks_by_serial = {
    "YOUR_SERIAL_0": mask0,  # shape HxW, bool or uint8
    "YOUR_SERIAL_1": mask1,
}
datapoints, scene_pcd, robot_pcd, robot_link_pcds = system.step(masks_by_serial=masks_by_serial)
```

For a fully custom stack (different robot type, no `TeleopPointCloudSystem`), start from **`SO101Leader`** / **`SO101Follower`** in LeRobot and **`SystemStateViewer`** in `lerobot_3d.point_clouds.system_vis`, passing a **`TeleopSystemConfig`** and calling **`update(*actions)`** with one dict per follower each tick.

For all CLI options:

```bash
lerobot-teleop -h
```
