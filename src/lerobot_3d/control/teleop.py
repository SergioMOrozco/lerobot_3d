from __future__ import annotations

import argparse
import contextlib
import time

import numpy as np
import open3d as o3d

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot_3d.common.types import Datapoint, RobotSnapshot
from lerobot_3d.teleop_config import TeleopSystemConfig, load_teleop_system_config
from lerobot_3d.point_clouds.system_vis import SystemStateViewer
from lerobot_3d.recording.dataset_recorder import LeRobotDatasetRecorder
from lerobot_3d.recording.episode_controller import EpisodeController, EpisodeState
from lerobot_3d.recording.keyboard import TerminalKeyListener
from lerobot_3d.teleoperators.yam_keyboard import KEYBOARD_HELP

_OVERRUN_WARN_INTERVAL_S = 5.0

_CLEAR_LINE = "\r\x1b[K"


class _StatusLine:
    """A terminal line rewritten in place (the recording timer); log lines print above it."""

    def __init__(self):
        self._text = ""

    def show(self, text: str) -> None:
        if text != self._text:
            print(_CLEAR_LINE + text, end="", flush=True)
            self._text = text

    def clear(self) -> None:
        if self._text:
            print(_CLEAR_LINE, end="", flush=True)
            self._text = ""

    def log(self, msg: str) -> None:
        self.clear()
        print(msg)


class TeleopPointCloudSystem:

    def __init__(self, config: TeleopSystemConfig):
        self.config = config
        if config.robot_type == "yam":
            from lerobot_3d.teleoperators.yam_keyboard import YamKeyboard, YamKeyboardConfig

            self.leaders = [
                YamKeyboard(YamKeyboardConfig(sides=config.yam_sides, urdf_path=config.urdf_path, id="yam_keyboard"))
                for _ in config.leaders
            ]
        else:
            self.leaders = [
                SO101Leader(SO101LeaderConfig(port=ax.port, id=ax.id)) for ax in config.leaders
            ]
        self.viewer = SystemStateViewer(config)
        if config.robot_type == "yam" and self.leaders:
            # IK in the same world as the arms drawn in viser (calibrated base poses included).
            self.leaders[0].config.base_poses = self.viewer.base_poses
            home_button = self.viewer.viewer.server.gui.add_button("Home", hint="Ease every arm back to its rest pose")
            # Clicks arrive on viser's thread; the control loop applies them in step().
            self._home_requested = False

            @home_button.on_click
            def _on_home(_) -> None:
                self._home_requested = True
        self.last_actions: list[dict] | None = None
        """Actions used by the most recent :meth:`step` (from the leaders or the caller)."""

    def connect(self) -> None:
        if not self.leaders:
            return
        print("Connecting devices...")
        for leader in self.leaders:
            leader.connect()
        if self.config.robot_type == "yam" and self.viewer.yam_followers:
            # Start the keyboard from where the real arms are, so the first command doesn't move them.
            for follower in self.viewer.followers:
                self.leaders[0].set_joints(follower.get_observation())
        print("Connected.")

    def step(
        self, action=None, masks_by_serial=None
    ) -> tuple[
        list[Datapoint],
        o3d.geometry.PointCloud,
        list[np.ndarray],
        list[dict[str, np.ndarray]],
        list[RobotSnapshot],
    ]:
        """One control cycle: read leaders (or use ``action``), update followers and viewer.

        Args:
            action: Optional per-follower action list (one motor-space dict per
                follower, positionally aligned with ``config.followers`` -- same
                shape ``SystemStateViewer.update()`` already expects). When given,
                it's sent to the followers instead of the leaders' teleop action,
                letting a caller (e.g. a computed IK target) drive the robot directly.
                Omit to teleop from the leaders. Required when no leaders are
                configured. With no leaders and no followers, pass one action per
                virtual robot (``config.num_robots``); each poses its own URDF in the
                viser grid instead of being sent to hardware.
            masks_by_serial: Optional mapping ``{camera_serial: mask}`` or sequence of
                masks aligned with ``config.realsense_serials``. Nonzero/True mask
                pixels are kept in the fused scene point cloud.

        Returns:
            datapoints: Raw per-camera datapoints used to build the fused point cloud.
            scene_pcd: Open3D fused scene point cloud in world frame, including colors.
            robot_pcds: per robot, ``(M, 3)`` ``float64`` sampled mesh points in that robot's
                base frame (with followers: one entry, the first follower).
            robot_link_pcds: per robot, clouds keyed by URDF link name (base frame).
            robot_snapshots: per robot, a :class:`RobotSnapshot` -- motor-space
                ``joint_positions``, ``joint_radians``, ``pcd``, ``link_pcds``,
                ``link_poses`` (base frame) and the viser grid ``base_offset``.

        With real YAM arms (followers) and no keyboard or ``action``, the arms are only read, and their
        measured joints become :attr:`last_actions` (what a dataset records as the action). With an
        action, :attr:`last_actions` is what the arms were actually sent, after their speed limits.

        Poll ``self.viewer.quit`` to know when to stop the outer loop, then call :meth:`close`.
        """
        if self.config.robot_type == "yam":
            return self._step_yam(action, masks_by_serial)
        if action is not None:
            actions = action
        elif not self.leaders:
            raise ValueError(
                "No leaders configured; call step(action) with one action dict per follower "
                "(or one per virtual robot, config.num_robots, with no followers)."
            )
        elif self.config.followers:
            actions = [leader.get_action() for leader in self.leaders]
        else:
            # URDF-only mode visualizes a single virtual robot, driven by the first leader.
            actions = [self.leaders[0].get_action()]
        self.last_actions = list(actions)
        return self.viewer.update(*actions, masks_by_serial=masks_by_serial)

    def _step_yam(self, action, masks_by_serial):
        if action is not None:
            actions = list(action)
        elif self.leaders:
            # One keyboard drives every arm; split its action into one per arm.
            if self._home_requested:
                self._home_requested = False
                self.leaders[0].home()
            merged = self.leaders[0].get_action()
            actions = [{key: merged[key] for key in state.keys} for state in self.viewer.robot_states]
        elif self.viewer.yam_followers:
            actions = []  # observe only
        else:
            raise ValueError("No YAM arms or keyboard: call step(action) with one action dict per virtual arm.")
        result = self.viewer.update(*actions, masks_by_serial=masks_by_serial)
        if not self.viewer.yam_followers:
            self.last_actions = actions
        elif actions:
            self.last_actions = self.viewer.sent_actions
        else:
            self.last_actions = [dict(snapshot.joint_positions) for snapshot in result[4]]
        return result

    def close(self) -> None:
        """Release cameras, robots, and recording resources."""
        self.viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop + point cloud viewer.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Teleop system config YAML (cwd, LEROBOT_3D_TELEOP_CONFIG, or src/ next to "
        "package). Omit to use the default teleop_config.yaml.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=60.0,
        help="Main loop rate in Hz (default 60). Use 0 or negative for no sleep (full speed). "
        "Ignored while dataset recording is configured (dataset_fps is used instead).",
    )

    args = parser.parse_args()

    config = load_teleop_system_config(args.config)
    if not config.leaders and config.robot_type != "yam":
        parser.exit(
            1,
            "No leaders configured: the CLI loop only teleops from leaders. Drive the system "
            "from your own script with TeleopPointCloudSystem.step(action) instead.\n",
        )
    recording = bool(config.dataset_repo_id)
    status_line = _StatusLine()
    controller = (
        EpisodeController(
            LeRobotDatasetRecorder(
                config.dataset_repo_id,
                root=config.dataset_root,
                fps=config.dataset_fps,
                task=config.dataset_task,
                robot_type="yam_follower" if config.robot_type == "yam" else "so101_follower",
            ),
            fps=config.dataset_fps,
            log=status_line.log,
        )
        if recording
        else None
    )

    system = TeleopPointCloudSystem(config)
    system.connect()
    keyboard = system.leaders[0] if config.robot_type == "yam" and system.leaders else None
    # YAM with no followers and no keyboard: nothing moves the arms, so show the URDF rest pose.
    preview_actions = (
        system.viewer.rest_actions()
        if config.robot_type == "yam" and not config.followers and keyboard is None
        else None
    )

    hz = config.dataset_fps if recording else args.hz
    period_s = None if hz is None or hz <= 0 else 1.0 / hz
    last_overrun_warning = float("-inf")
    try:
        with TerminalKeyListener() if recording or keyboard else contextlib.nullcontext() as keys:
            if keyboard is not None:
                status_line.log(f"[keyboard] {KEYBOARD_HELP}")
            if controller is not None:
                controller.print_help()
            while not system.viewer.quit:
                t_iter_start = time.monotonic()
                # The keyboard teleop takes its keys before the step; the rest go to the recorder.
                pressed = keys.get_keys() if keys is not None else []
                if keyboard is not None:
                    pressed = keyboard.handle_keys(pressed)
                datapoints, _scene_pcd, _robot_pcds, _robot_link_pcds, snapshots = system.step(preview_actions)

                if controller is not None:
                    was_recording = controller.state is EpisodeState.RECORDING
                    controller.on_step(datapoints, snapshots, system.last_actions)
                    # Only the step + frame capture set the recorded rate; saving an episode
                    # (video encoding) happens between episodes and is excluded.
                    capture_elapsed = time.monotonic() - t_iter_start
                    for key in pressed:
                        controller.handle_key(key)
                    system.viewer.viewer.set_status(controller.status_text())
                    if controller.state is EpisodeState.RECORDING:
                        status_line.show(controller.status_text())
                    if (
                        was_recording
                        and period_s is not None
                        and capture_elapsed > period_s
                        and t_iter_start - last_overrun_warning > _OVERRUN_WARN_INTERVAL_S
                    ):
                        last_overrun_warning = t_iter_start
                        status_line.log(
                            f"[recorder] Warning: step took {capture_elapsed * 1000:.0f} ms, over "
                            f"the {period_s * 1000:.0f} ms budget for dataset_fps={hz}; recorded "
                            "timestamps assume a fixed rate. Lower dataset_fps or camera settings."
                        )

                if keyboard is not None:
                    status = keyboard.status_text()
                    if controller is not None:
                        status += "  |  " + controller.status_text()
                    system.viewer.viewer.set_status(status)

                if period_s is not None:
                    elapsed = time.monotonic() - t_iter_start
                    time.sleep(max(0.0, period_s - elapsed))
    finally:
        status_line.clear()
        try:
            if controller is not None:
                controller.close()
        finally:
            system.close()


if __name__ == "__main__":
    main()
