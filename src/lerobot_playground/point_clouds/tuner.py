import threading
import tkinter as tk
from tkinter import ttk


class StateTuner:
    """Runs Tkinter GUI in a background thread for quit/capture/save controls."""
    def __init__(self):

        self.quit = False
        self.capture = False
        self.save_subgoal = False

        self.thread = None
        self.root = None

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def start(self):
        """Launch the GUI in a background thread (non-blocking)."""
        self.thread = threading.Thread(target=self._run_gui, daemon=True)
        self.thread.start()

    def get_state(self):
        """Return control flags/state for callers that want a polling API."""
        return {}

    # ---------------------------------------------------------
    # Internal GUI setup
    # ---------------------------------------------------------
    def _run_gui(self):
        self.root = tk.Tk()
        self.root.title("State Tuner")

        # --- Quit Button ---
        quit_button = ttk.Button(self.root, text="Quit", command=self._quit_gui)
        quit_button.pack(pady=10)

        # --- capture Button ---
        capture_button = ttk.Button(self.root, text="Capture", command=self._capture)
        capture_button.pack(pady=10)

        ttk.Label(self.root, text="Press 's' in this window to save fused scene → subgoals/N.npz").pack(
            pady=(0, 6)
        )
        self.root.bind_all("<KeyPress-s>", self._on_save_subgoal_key)
        self.root.bind_all("<KeyPress-S>", self._on_save_subgoal_key)

        self.root.mainloop()

    def _quit_gui(self):
        self.quit = True

    def _capture(self):
        self.capture = True

    def _on_save_subgoal_key(self, event=None):
        self.save_subgoal = True

if __name__ == "__main__":
    tuner = StateTuner()

    tuner.start()   # <-- non-blocking

    while True:
        state = tuner.get_state()
        print(state)