# lerobot_playground

SO101 multi-camera teleop, fused point clouds, Foxglove logging, and related calibration utilities.

## Install

**Editable** (recommended while developing):

```bash
cd /path/to/lerobot_playground
pip install -e .
```

**Into another environment** from a checkout or wheel:

```bash
pip install /path/to/lerobot_playground
```

**RealSense** cameras are required for teleop; install the extra so `pyrealsense2` is available:

```bash
pip install -e ".[realsense]"
```

Other optional extras (see `pyproject.toml`): `viser`.

Import the package in Python as `lerobot_playground`, for example:

```python
from lerobot_playground.paths import CALIBRATION_DIR
```

Other console entry points from this repo include `lerobot-flow-solver` and `lerobot-flow-solver-so101`. Those write artifacts under the current working directory unless you set **`LEROBOT_PLAYGROUND_ARTIFACT_DIR`**.

## Teleop (`lerobot-teleop`)

Teleop drives **N SO101 follower** arms from **N SO101 leader** teleoperators while streaming **one or more Intel RealSense** cameras, fusing depth into a scene point cloud, sampling the first follower’s URDF for a robot point cloud, and logging to **Foxglove** (a server is started automatically).

### What you need

- `pyrealsense2` (use `pip install -e ".[realsense]"`).
- **Camera extrinsics** in JSON (see below). Intrinsics can be written on shutdown when missing (see `SystemStateViewer.close`).
- A **`TeleopSystemConfig`** (see `lerobot_playground.hardware_config`) listing **RealSense serials**, **leader** `port` / `id`, and **follower** `port` / `id`. Defaults match the original lab two-arm layout; override in Python or use CLI overrides below.

### Calibration files

Extrinsics are **not** bundled in the package. Resolution order for the default filename `extrinsic_calibration.json`:

1. Environment variable **`LEROBOT_PLAYGROUND_EXTRINSIC_JSON`** (absolute path to the file).
2. File in the **current working directory**.
3. In a dev checkout, **`src/extrinsic_calibration.json`** next to the installed package tree.

You can always pass an explicit path:

```bash
lerobot-teleop --extrinsic-json /path/to/extrinsic_calibration.json
```

### How to run

```bash
lerobot-teleop
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--hz 60` | Main loop rate in Hz (default `60`). Use `0` or negative for no pacing. |
| `--recording_name NAME` | If non-empty, records RGB/depth per camera and gripper points under `recordings/NAME/`. |
| `--extrinsic-json PATH` | Camera extrinsics JSON (see above). |
| `--realsense-serial SERIAL` | Repeat once per camera (order must match keys in the extrinsics JSON). If omitted, uses defaults from `TeleopSystemConfig`. |
| `--no-tune` | Disable the Tk bbox/capture tuner. |
| `--visualization foxglove` | `foxglove`, `open3d`, `both`, or `none`. `open3d` displays the full fused scene cloud plus the robot cloud in `point_cloud_viewer`. |

Examples:

```bash
# Default rate, no recording
lerobot-teleop

# Record a session
lerobot-teleop --recording_name my_run

# Slower loop, explicit extrinsics
lerobot-teleop --hz 10 --extrinsic-json ./my_extrinsics.json

# Two cameras with explicit serials
lerobot-teleop --realsense-serial 111 --realsense-serial 222

# Skip Foxglove and show the full scene + robot clouds in Open3D
lerobot-teleop --visualization open3d
```

With `--visualization foxglove` or `both`, open **Foxglove** and connect to the websocket URL printed in the console to view fused point clouds, transforms, and the robot cloud. With `--visualization open3d` or `both`, an Open3D `point_cloud_viewer` window displays the full fused scene point cloud and overlays the robot cloud in red. Quit from the tuner UI or your usual session flow as implemented in `StateTuner`.

### Custom teleop script

Build a **`TeleopSystemConfig`** (`lerobot_playground.hardware_config`) with **`SO101AxisConfig`** entries for each **leader** and **follower** (`port` + LeRobot `id`), **`realsense_serials`**, and optional fields (`urdf_path`, `robot_calibration_ids`, `tune`, `publish_to_foxglove`, `display_point_cloud_viewer`). **`len(leaders)` must equal `len(followers)`** (one leader action per follower). **`robot_calibration_ids`** defaults to each follower’s `id`; the **first** follower’s observation drives the mesh / TF visualization returned as **`robot_pcd`** and **`robot_link_pcds`**. Robot mesh points are sampled once at startup and then transformed by FK every step.

**Minimal** (same behavior as the CLI defaults, but from your own file): call **`step()`** each tick for **`scene_pcd`**, **`robot_pcd`**, and **`robot_link_pcds`**. `scene_pcd` / `robot_pcd` are **`(N, 3)`** / **`(M, 3)`** **`float64`** arrays; `robot_link_pcds` is a **`dict[str, np.ndarray]`** keyed by URDF link name. Call **`close()`** when `viewer.quit` is set:

```python
import time
from dataclasses import replace

from lerobot_playground.control.teleop import TeleopPointCloudSystem
from lerobot_playground.hardware_config import SO101AxisConfig, TeleopSystemConfig

if __name__ == "__main__":
    hz = 15.0
    period_s = None if hz <= 0 else 1.0 / hz

    config = replace(
        TeleopSystemConfig(),
        realsense_serials=("YOUR_SERIAL_0", "YOUR_SERIAL_1"),
        extrinsic_json="extrinsic_calibration.json",
        recording_name="",
        publish_to_foxglove=False,
        display_point_cloud_viewer=True,
        leaders=(
            SO101AxisConfig("/dev/ttyACM0", "bender_leader_arm"),
            SO101AxisConfig("/dev/ttyACM1", "clamps_leader_arm"),
        ),
        followers=(
            SO101AxisConfig("/dev/ttyACM3", "bender_follower_arm"),
            SO101AxisConfig("/dev/ttyACM2", "clamps_follower_arm"),
        ),
    )

    system = TeleopPointCloudSystem(config)
    system.connect()
    try:
        while not system.viewer.quit:
            t0 = time.monotonic()
            scene_pcd, robot_pcd, robot_link_pcds = system.step()
            # use scene_pcd / robot_pcd / robot_link_pcds here (e.g. policy, logging)
            if period_s is not None:
                time.sleep(max(0.0, period_s - (time.monotonic() - t0)))
    finally:
        system.close()
```

For a fully custom stack (different robot type, no `TeleopPointCloudSystem`), start from **`SO101Leader`** / **`SO101Follower`** in LeRobot and **`SystemStateViewer`** in `lerobot_playground.point_clouds.system_vis`, passing a **`TeleopSystemConfig`** and calling **`update(*actions)`** with one dict per follower each tick.

For all CLI options:

```bash
lerobot-teleop -h
```
