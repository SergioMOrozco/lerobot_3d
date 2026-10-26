"""Click-to-segment the robot with SAM2 and save ``calibration_files/<serial>/mask.png``.

``icp.py`` reads the mask's **alpha channel** (opaque = robot, transparent = background), so
:func:`save_mask_png` writes the color image as an RGBA cutout in that convention.

SAM2 is an optional dependency (``pip install -e ".[segment]"``); it's only imported when
:class:`RobotSegmenter` first predicts.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

def segment_help(target: str = "robot") -> str:
    return f"L-click: {target}  R-click: background  Enter: save  u: undo  r: reset  s/Esc: skip"


SEGMENT_HELP = segment_help()

_KEY_ENTER = (13, 10)
_KEY_SPACE = 32
_KEY_ESC = 27
_KEY_BACKSPACE = 8


@dataclass
class PromptPoints:
    """SAM point prompts: pixel ``(x, y)`` coordinates with label 1 (robot) or 0 (background)."""

    points: list[tuple[int, int]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def add(self, x: int, y: int, positive: bool) -> None:
        self.points.append((int(x), int(y)))
        self.labels.append(1 if positive else 0)

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.labels.pop()

    def reset(self) -> None:
        self.points.clear()
        self.labels.clear()

    def __len__(self) -> int:
        return len(self.points)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(N, 2)`` float32 coordinates and ``(N,)`` int32 labels, as SAM2 expects."""
        coords = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        return coords, np.asarray(self.labels, dtype=np.int32)


def save_mask_png(path, color_bgr: np.ndarray, mask: np.ndarray) -> None:
    """Write ``color_bgr`` as an RGBA PNG whose alpha is 255 on ``mask`` and 0 elsewhere."""
    color_bgr = np.asarray(color_bgr)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != color_bgr.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {color_bgr.shape[:2]}")
    bgra = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), bgra):
        raise OSError(f"Failed to write {path}")


class RobotSegmenter:
    """SAM2 image predictor, loaded on first use and reused across captures."""

    def __init__(self, model_id: str = DEFAULT_SAM2_MODEL_ID):
        self.model_id = model_id
        self._predictor = None
        self._torch = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                'SAM2 is not installed; run `pip install -e ".[segment]"` to enable '
                "click-to-segment on Capture."
            ) from e
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[capture] Loading SAM2 ({self.model_id}) on {self._device}...")
        self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id, device=self._device)

    def _autocast(self):
        if self._device == "cuda":
            return self._torch.autocast("cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()

    def set_image(self, color_bgr: np.ndarray) -> None:
        """Compute (and cache) the image embedding for ``color_bgr``."""
        self._load()
        rgb = cv2.cvtColor(np.asarray(color_bgr), cv2.COLOR_BGR2RGB)
        with self._torch.inference_mode(), self._autocast():
            self._predictor.set_image(rgb)

    def predict(self, prompts: PromptPoints) -> np.ndarray:
        """``(H, W)`` bool robot mask for the current image and point prompts."""
        coords, labels = prompts.as_arrays()
        # A single click is ambiguous (part vs. whole); let SAM propose several and keep the best.
        multimask = len(prompts) == 1
        with self._torch.inference_mode(), self._autocast():
            masks, scores, _ = self._predictor.predict(
                point_coords=coords, point_labels=labels, multimask_output=multimask
            )
        return np.asarray(masks[int(np.argmax(scores))]) > 0


def _render(color_bgr: np.ndarray, mask: np.ndarray | None, prompts: PromptPoints, help_text: str) -> np.ndarray:
    canvas = color_bgr.copy()
    if mask is not None:
        tint = np.zeros_like(canvas)
        tint[mask] = (0, 255, 0)
        canvas[mask] = cv2.addWeighted(canvas, 0.5, tint, 0.5, 0)[mask]
    for (x, y), label in zip(prompts.points, prompts.labels):
        cv2.circle(canvas, (x, y), 6, (0, 255, 0) if label else (0, 0, 255), -1)
        cv2.circle(canvas, (x, y), 6, (255, 255, 255), 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(canvas, help_text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas


def segment_interactively(
    segmenter: RobotSegmenter, color_bgr: np.ndarray, window_name: str, target: str = "robot"
) -> np.ndarray | None:
    """Show ``color_bgr`` in an OpenCV window and refine a SAM2 mask of ``target`` from clicks.

    Returns the ``(H, W)`` bool mask on Enter/Space, or ``None`` if the user skips (s/Esc,
    closing the window, or accepting with no points).
    """
    color_bgr = np.asarray(color_bgr)
    segmenter.set_image(color_bgr)
    prompts = PromptPoints()
    state = {"mask": None, "dirty": True}

    def _update_mask() -> None:
        state["mask"] = segmenter.predict(prompts) if len(prompts) else None
        state["dirty"] = True

    def _on_mouse(event, x, y, _flags, _param) -> None:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            prompts.add(x, y, positive=event == cv2.EVENT_LBUTTONDOWN)
            _update_mask()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _on_mouse)
    try:
        while True:
            if state["dirty"]:
                cv2.imshow(window_name, _render(color_bgr, state["mask"], prompts, segment_help(target)))
                state["dirty"] = False
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in _KEY_ENTER or key == _KEY_SPACE:
                return state["mask"]
            if key in (ord("s"), _KEY_ESC):
                return None
            if key in (ord("u"), _KEY_BACKSPACE):
                prompts.undo()
                _update_mask()
            elif key == ord("r"):
                prompts.reset()
                _update_mask()
    finally:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
