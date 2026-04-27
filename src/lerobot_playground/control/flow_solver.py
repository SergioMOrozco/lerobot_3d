#!/usr/bin/env python3
"""
Replay SO101 follower joint trajectories (motor-space ``send_action`` dicts).

**Recording** Jacobian / flow trajectories for the arm is done in ``flow_solver_so101.py``
(Viser sliders → ``recorded_flow.npy`` → Solve → ``recorded_flow_robot.npz``). This script
is for **playing** that ``.npz`` on hardware, or for logging raw teleop with ``record``.

Uses the same keys as ``SO101Follower.get_observation()`` (``shoulder_pan.pos``, …).

After ``pip install lerobot-playground`` (or ``pip install -e .`` from the repo):

  python -m lerobot_playground.control.flow_solver play --input recorded_flow_robot.npz --hz 15
  python -m lerobot_playground.control.flow_solver record --out teleop_log.npz --hz 15
"""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# Same motor keys as used across this repo (lerobot observation / action)
JOINT_KEYS: list[str] = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Presets aligned with ``lerobot_playground.point_clouds.system_vis`` (robot_1 / robot_2)
ROBOT_PRESETS: dict[str, tuple[str, str]] = {
    "bender": ("bender_follower_arm", "/dev/ttyACM3"),
    "clamps": ("clamps_follower_arm", "/dev/ttyACM2"),
}


def _obs_to_motor_row(obs: dict) -> np.ndarray:
    row = np.empty(len(JOINT_KEYS), dtype=np.float64)
    for i, k in enumerate(JOINT_KEYS):
        v = obs[k]
        row[i] = float(np.asarray(v).item())
    return row


def _motor_row_to_action(row: np.ndarray) -> dict:
    return {k: float(row[i]) for i, k in enumerate(JOINT_KEYS)}


def _connect(robot_id: str, port: str | None) -> SO101Follower:
    rid, dev = ROBOT_PRESETS[robot_id]
    if port:
        dev = port
    robot = SO101Follower(SO101FollowerConfig(port=dev, id=rid))
    print(f"Connecting {rid} on {dev}...")
    robot.connect()
    print("Connected.")
    return robot


def print_current_state(robot: SO101Follower) -> None:
    obs = robot.get_observation()
    print("[state] motor observation:")
    for k in JOINT_KEYS:
        print(f"  {k}: {float(np.asarray(obs[k]).item()):.4f}")


def cmd_record(args: argparse.Namespace) -> None:
    robot = _connect(args.robot, args.port)
    print_current_state(robot)

    hz = float(args.hz)
    period = None if hz <= 0 else 1.0 / hz
    buf: list[np.ndarray] = []

    stop = {"flag": False}

    def _on_sigint(_sig, _frame):
        stop["flag"] = True
        print("\n[record] stopping (signal or end of duration)...", flush=True)

    signal.signal(signal.SIGINT, _on_sigint)

    duration = float(args.duration)
    t_end = time.monotonic() + duration if duration > 0 else None

    print(
        "[record] Recording motor observations at {:.2f} Hz. ".format(hz if hz > 0 else float("inf"))
        + ("Stop with Ctrl+C or when duration expires.\n" if duration <= 0 else f"Stopping after {duration} s.\n")
    )

    try:
        while not stop["flag"]:
            if t_end is not None and time.monotonic() >= t_end:
                break
            t0 = time.monotonic()
            obs = robot.get_observation()
            buf.append(_obs_to_motor_row(obs))
            if period is not None:
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, period - elapsed))
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    motor = np.stack(buf, axis=0) if buf else np.zeros((0, len(JOINT_KEYS)), dtype=np.float64)
    meta = np.array(JOINT_KEYS, dtype=object)
    np.savez_compressed(out, motor=motor, joint_keys=meta, hz=np.float64(hz))
    print(f"[record] Saved {motor.shape[0]} frames to {out}")


def cmd_play(args: argparse.Namespace) -> None:
    path = Path(args.input).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    motor = np.asarray(data["motor"], dtype=np.float64)
    if motor.ndim != 2 or motor.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"Expected motor with shape (T, {len(JOINT_KEYS)}), got {motor.shape}")

    if "joint_keys" in data:
        joint_keys = [str(x) for x in data["joint_keys"].tolist()]
        if joint_keys != JOINT_KEYS:
            print("[play] Warning: joint_keys in file differ from defaults; using file order.")
    else:
        joint_keys = JOINT_KEYS

    if motor.shape[1] != len(joint_keys):
        raise ValueError("motor width does not match joint_keys")

    robot = _connect(args.robot, args.port)
    print_current_state(robot)

    hz = float(args.hz)
    period = None if hz <= 0 else 1.0 / hz
    n = motor.shape[0]
    loops = int(args.loops)
    print(f"[play] Playing {n} frames x {loops} loop(s) at {hz} Hz (0 = no sleep).")

    for loop_i in range(loops):
        for t in range(n):
            t0 = time.monotonic()
            row = motor[t]
            action = {joint_keys[i]: float(row[i]) for i in range(len(joint_keys))}

            print(action[joint_keys[-1]])

            if action[joint_keys[-1]] < 20 and joint_keys[-1] == 'gripper.pos':
                action[joint_keys[-1]] = 0

            robot.send_action(action)
            if period is not None:
                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, period - elapsed))
        if loops > 1:
            print(f"[play] completed loop {loop_i + 1}/{loops}")

    print("[play] Done.")
    print_current_state(robot)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record / replay SO101 follower joint trajectories.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Stream get_observation() to an .npz file")
    pr.add_argument("--out", type=str, required=True, help="Output .npz path")
    pr.add_argument("--robot", type=str, default="bender", choices=list(ROBOT_PRESETS))
    pr.add_argument("--port", type=str, default=None, help="Override serial port (e.g. /dev/ttyACM3)")
    pr.add_argument("--hz", type=float, default=15.0, help="Sampling rate (default 15). 0 = no sleep.")
    pr.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to record; 0 = until Ctrl+C",
    )
    pr.set_defaults(func=cmd_record)

    pp = sub.add_parser("play", help="Replay a trajectory saved by `record`")
    pp.add_argument("--input", type=str, required=True, help="Input .npz from record")
    pp.add_argument("--robot", type=str, default="bender", choices=list(ROBOT_PRESETS))
    pp.add_argument("--port", type=str, default=None)
    pp.add_argument("--hz", type=float, default=15.0, help="Playback rate (default 15). 0 = no sleep.")
    pp.add_argument("--loops", type=int, default=1, help="Repeat trajectory this many times")
    pp.set_defaults(func=cmd_play)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
