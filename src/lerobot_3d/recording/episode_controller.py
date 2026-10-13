"""Enter / Space / y-n episode control on top of :class:`LeRobotDatasetRecorder`.

``idle --Enter--> recording --Space--> confirming --y/n--> idle``. The control loop keeps
running in every state; frames are only added while ``recording``.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

ENTER_KEYS = ("\n", "\r")
SPACE_KEY = " "


def format_duration(seconds: float) -> str:
    """``MM:SS.s`` (minutes keep counting past 59)."""
    minutes, secs = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{secs:04.1f}"


class EpisodeState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    CONFIRMING = "confirming"


class EpisodeController:
    def __init__(self, recorder, fps: float, log: Callable[[str], None] = print):
        self.recorder = recorder
        self.fps = fps
        self.log = log
        self.state = EpisodeState.IDLE
        self.num_frames = 0
        """Frames added to the current episode (reset when a new one starts)."""

    @property
    def elapsed_s(self) -> float:
        """Recorded duration of the current episode, in dataset time (frames / fps)."""
        return self.num_frames / self.fps

    def status_text(self) -> str:
        """One-line status for live display, e.g. ``● REC episode 3  01:12.4``."""
        episode = self.recorder.num_saved_episodes
        if self.state is EpisodeState.RECORDING:
            return f"● REC episode {episode}  {format_duration(self.elapsed_s)}"
        if self.state is EpisodeState.CONFIRMING:
            return f"Episode {episode} stopped at {format_duration(self.elapsed_s)} -- keep? [y/n]"
        return f"Idle -- {episode} episodes saved (Enter to record)"

    def print_help(self) -> None:
        self.log(
            f"[recorder] {self.recorder.num_saved_episodes} episodes saved. "
            "Press Enter to start an episode, Space to end it."
        )

    def handle_key(self, key: str) -> None:
        if self.state is EpisodeState.IDLE and key in ENTER_KEYS:
            self.recorder.start_episode()
            self.num_frames = 0
            self.state = EpisodeState.RECORDING
            self.log(
                f"[recorder] ● Recording episode {self.recorder.num_saved_episodes} "
                "(Space to end)..."
            )
        elif self.state is EpisodeState.RECORDING and key == SPACE_KEY:
            n_frames = self.recorder.stop_episode()
            if n_frames == 0:
                self.recorder.discard_episode()
                self.state = EpisodeState.IDLE
                self.log("[recorder] Episode had no frames; discarded.")
                self.print_help()
                return
            self.state = EpisodeState.CONFIRMING
            self.log(
                f"[recorder] Episode {self.recorder.num_saved_episodes}: {n_frames} frames "
                f"({n_frames / self.fps:.1f} s). Keep? [y/n]"
            )
        elif self.state is EpisodeState.CONFIRMING and key.lower() in ("y", "n"):
            if key.lower() == "y":
                self.log("[recorder] Saving episode (encoding video)...")
                self.recorder.save_episode()
                self.log("[recorder] Saved.")
            else:
                self.recorder.discard_episode()
                self.log("[recorder] Discarded.")
            self.state = EpisodeState.IDLE
            self.print_help()

    def on_step(self, datapoints, snapshots, actions) -> None:
        """Add this step while recording; ``snapshots``/``actions`` are per robot (or one robot's)."""
        if self.state is EpisodeState.RECORDING:
            self.recorder.add_frame(datapoints, snapshots, actions)
            self.num_frames += 1

    def close(self) -> None:
        """Discard any unconfirmed episode and finalize the dataset."""
        if self.state is not EpisodeState.IDLE:
            self.log("[recorder] Quitting mid-episode; discarding the unsaved episode.")
            self.recorder.discard_episode()
            self.state = EpisodeState.IDLE
        self.recorder.finalize()
