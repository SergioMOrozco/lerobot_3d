"""``lerobot-trim``: cut the idle start/end off recorded episodes, choosing the cut points by eye.

With no ``--keep``/``--drop`` it opens a viser page: scrub or play each episode with every camera
shown, mark the first and last frame to keep (or drop the whole episode), then save. Saving writes
a **new** dataset; the source is never modified. Kept frames are copied as-is: actions, state,
depth (bit-exact), intrinsics/extrinsics and link clouds. RGB video is re-encoded, so it goes
through one more lossy encode. The sidecar is copied, with a ``trim`` entry recording which source
frames each new episode came from.

    lerobot-trim --repo-id Sorozco0612/tblock_demos                  # pick cut points in viser
    lerobot-trim --repo-id Sorozco0612/tblock_demos --keep 0:24:81   # episode 0, frames 24..81
"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from lerobot_3d.recording.dataset_recorder import RGB_PREFIX, SIDECAR_NAME

_DECODE_LOCK = threading.Lock()
"""LeRobot's torchcodec video decoding segfaults (in libdav1d) when two threads decode at once,
e.g. the UI scrubbing while a save runs. Every dataset read here goes through :func:`read_frame`."""


def read_frame(dataset, index: int) -> dict:
    """``dataset[index]``, one decode at a time across threads."""
    with _DECODE_LOCK:
        return dataset[index]


Ranges = dict[int, "tuple[int, int] | None"]
"""Per source episode: inclusive ``(first, last)`` frame to keep, or ``None`` to drop it.
Episodes not listed are kept whole."""


def parse_ranges(keep: list[str] | None, drop: list[int] | None) -> Ranges:
    """``["EP:FIRST:LAST", ...]`` and ``[EP, ...]`` -> :data:`Ranges`."""
    ranges: Ranges = {}
    for spec in keep or []:
        parts = spec.split(":")
        if len(parts) != 3 or not all(p.strip().lstrip("-").isdigit() for p in parts):
            raise ValueError(f"--keep expects EPISODE:FIRST:LAST (inclusive frames), got {spec!r}")
        ep, first, last = (int(p) for p in parts)
        if ep in ranges:
            raise ValueError(f"Episode {ep} given more than once")
        ranges[ep] = (first, last)
    for ep in drop or []:
        if ep in ranges:
            raise ValueError(f"Episode {ep} is both kept and dropped")
        ranges[ep] = None
    return ranges


def resolve_ranges(lengths: dict[int, int], ranges: Ranges) -> dict[int, tuple[int, int] | None]:
    """Validate ``ranges`` against episode lengths and fill in whole episodes for unlisted ones."""
    unknown = sorted(set(ranges) - set(lengths))
    if unknown:
        raise ValueError(f"Episodes {unknown} not in the dataset (it has {sorted(lengths)})")
    resolved = {}
    for ep, n in lengths.items():
        span = ranges.get(ep, (0, n - 1))
        if span is not None:
            first, last = span
            if not 0 <= first <= last < n:
                raise ValueError(f"Episode {ep}: frames {first}..{last} must satisfy 0 <= first <= last <= {n - 1}")
        resolved[ep] = span
    if all(span is None for span in resolved.values()):
        raise ValueError("Every episode is dropped; nothing to write")
    return resolved


def _resolve_root(repo_id: str, root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return (Path(HF_LEROBOT_HOME) / repo_id).resolve()


def episode_lengths(dataset) -> dict[int, int]:
    episodes = dataset.meta.episodes
    return dict(zip(episodes["episode_index"], episodes["length"]))


def _episode_start(dataset, episode: int) -> int:
    """Global row of an episode's first frame."""
    episodes = dataset.meta.episodes
    return int(episodes["dataset_from_index"][list(episodes["episode_index"]).index(episode)])


def _frame_for_writing(item: dict, features: dict) -> dict:
    """A loaded item back into ``add_frame`` form. Images/videos load as float ``[0, 1]`` CHW from
    uint8 sources, so ``rint(x * 255)`` recovers them exactly (packed depth stays bit-exact)."""
    frame = {"task": item["task"]}
    for key, feature in features.items():
        value = item[key]
        value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if feature["dtype"] in ("image", "video"):
            frame[key] = np.clip(np.rint(np.moveaxis(value, 0, -1) * 255.0), 0, 255).astype(np.uint8)
        else:
            frame[key] = value.astype(feature["dtype"]).reshape(feature["shape"])
    return frame


def trim_dataset(
    repo_id: str,
    root: str | Path | None,
    output_repo_id: str,
    output_root: str | Path | None,
    ranges: Ranges,
    progress=print,
) -> Path:
    """Write a copy of the dataset keeping only the frames in ``ranges``; returns its root.

    The copy is built in ``<output>.partial`` and moved into place only once complete, so a failed
    run never leaves a half-written dataset at the output path.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import DEFAULT_FEATURES

    source = _resolve_root(repo_id, root)
    target = _resolve_root(output_repo_id, output_root)
    if target == source:
        raise ValueError("Output must differ from the source dataset; trimming never edits it in place")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"{target} already exists; choose a new output repo id or root")
    src = LeRobotDataset(repo_id, root=source)
    spans = resolve_ranges(episode_lengths(src), ranges)

    features = {
        key: {k: v for k, v in feature.items() if k != "info"}
        for key, feature in src.meta.features.items() if key not in DEFAULT_FEATURES
    }
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    n_cameras = sum(key.startswith(RGB_PREFIX) for key in features)
    dst = LeRobotDataset.create(output_repo_id, src.fps, features, root=staging, robot_type=src.meta.robot_type,
                                use_videos=True, image_writer_threads=4 * max(n_cameras, 1))
    record = []
    try:
        for ep, span in spans.items():
            if span is None:
                progress(f"episode {ep}: dropped")
                continue
            first, last = span
            start = _episode_start(src, ep)
            for frame_index in range(first, last + 1):
                dst.add_frame(_frame_for_writing(read_frame(src, start + frame_index), features))
            dst.save_episode()
            record.append({"episode_index": len(record), "source_episode": ep,
                           "source_first_frame": first, "source_last_frame": last})
            progress(f"episode {ep}: kept frames {first}..{last} ({last - first + 1}) -> episode {len(record) - 1}")
        dst.finalize()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    sidecar_path = source / "meta" / SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    sidecar["trim"] = {"source_repo_id": repo_id, "episodes": record}
    (staging / "meta" / SIDECAR_NAME).write_text(json.dumps(sidecar, indent=2))
    shutil.rmtree(staging / "images", ignore_errors=True)  # LeRobot's temporary frame dumps
    if target.is_dir():
        target.rmdir()  # an empty leftover folder (checked above), e.g. from a crashed run
    staging.rename(target)
    return target


# ------------------------------------------------------------------------------ viser UI


def _mosaic(images: list[tuple[str, np.ndarray]], tile_width: int, info: list[str], cut: bool) -> np.ndarray:
    """Cameras in a grid (plus one cell of status text when there's room); cut frames are dimmed."""
    import cv2

    h0, w0 = images[0][1].shape[:2]
    tile_h = int(round(tile_width * h0 / w0))
    cols = int(np.ceil(np.sqrt(len(images) + 1)))
    rows = int(np.ceil((len(images) + 1) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_width, 3), np.uint8)
    for i, (name, image) in enumerate(images):
        tile = cv2.resize(image, (tile_width, tile_h), interpolation=cv2.INTER_AREA)
        if cut:
            tile = (tile * 0.35).astype(np.uint8)
        cv2.putText(tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        r, c = divmod(i, cols)
        canvas[r * tile_h:(r + 1) * tile_h, c * tile_width:(c + 1) * tile_width] = tile
    r, c = divmod(len(images), cols)
    x, y = c * tile_width + 16, r * tile_h + 44
    for j, line in enumerate(info):
        color = (255, 90, 90) if (cut and j == 1) else (120, 255, 120) if j == 1 else (230, 230, 230)
        cv2.putText(canvas, line, (x, y + 38 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return canvas


def trim_ui(args) -> None:
    import viser
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = _resolve_root(args.repo_id, args.root)
    src = LeRobotDataset(args.repo_id, root=source)
    lengths = episode_lengths(src)
    spans: dict[int, list[int] | None] = {ep: [0, n - 1] for ep, n in lengths.items()}
    cameras = [key[len(RGB_PREFIX):] for key in src.meta.video_keys]

    @lru_cache(maxsize=96)
    def camera_images(ep: int, frame_index: int) -> tuple:
        item = read_frame(src, _episode_start(src, ep) + frame_index)
        out = []
        for name in cameras:
            image = item[f"{RGB_PREFIX}{name}"].numpy()
            out.append((name, np.ascontiguousarray((np.moveaxis(image, 0, -1) * 255).astype(np.uint8))))
        return tuple(out)

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_width="large")
    gui = server.gui
    episode = gui.add_dropdown("Episode", tuple(str(ep) for ep in lengths), initial_value=str(next(iter(lengths))))
    frame = gui.add_slider("Frame", min=0, max=max(lengths[int(episode.value)] - 1, 1), step=1, initial_value=0)
    with gui.add_folder("Playback"):
        play = gui.add_button("Play / pause")
        speed = gui.add_slider("Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0)
        loop_kept = gui.add_checkbox("Play only kept frames", initial_value=False)
    with gui.add_folder("Cut"):
        keep = gui.add_multi_slider("Keep frames", min=0, max=max(lengths[int(episode.value)] - 1, 1), step=1,
                                    initial_value=(0, lengths[int(episode.value)] - 1))
        set_first = gui.add_button("First kept frame = current")
        set_last = gui.add_button("Last kept frame = current")
        drop = gui.add_checkbox("Drop this episode", initial_value=False)
        reset = gui.add_button("Keep whole episode")
    with gui.add_folder("Save"):
        output = gui.add_text("Output repo id", initial_value=args.output_repo_id or f"{args.repo_id}_trimmed")
        save = gui.add_button("Save trimmed copy", color="green")
    summary = gui.add_markdown("")

    state = {"playing": False, "saving": False, "syncing": False, "status": ""}
    dirty = threading.Event()

    def current() -> int:
        return int(episode.value)

    def sync_controls() -> None:
        """Point the sliders at the selected episode's length and stored cut."""
        ep, n = current(), lengths[current()]
        state["syncing"] = True
        try:
            frame.max = keep.max = max(n - 1, 1)
            frame.value = min(int(frame.value), n - 1)
            span = spans[ep]
            drop.value = span is None
            keep.value = tuple(span) if span is not None else (0, n - 1)
            keep.disabled = set_first.disabled = set_last.disabled = span is None
        finally:
            state["syncing"] = False
        dirty.set()

    def store_keep() -> None:
        if state["syncing"] or spans[current()] is None:
            return
        first, last = (int(v) for v in keep.value)
        spans[current()] = [min(first, last), max(first, last)]
        dirty.set()

    def set_bound(which: int) -> None:
        span = spans[current()]
        if span is None:
            return
        span[which] = int(frame.value)
        if span[0] > span[1]:  # moved past the other end: that end follows
            span[1 - which] = span[which]
        sync_controls()

    episode.on_update(lambda _: sync_controls())
    frame.on_update(lambda _: dirty.set())
    keep.on_update(lambda _: store_keep())
    set_first.on_click(lambda _: set_bound(0))
    set_last.on_click(lambda _: set_bound(1))

    @drop.on_update
    def _(_) -> None:
        if state["syncing"]:
            return
        spans[current()] = None if drop.value else [0, lengths[current()] - 1]
        sync_controls()

    @reset.on_click
    def _(_) -> None:
        spans[current()] = [0, lengths[current()] - 1]
        sync_controls()

    @play.on_click
    def _(_) -> None:
        state["playing"] = not state["playing"]

    @save.on_click
    def _(_) -> None:
        if state["saving"]:
            return
        state["saving"] = True
        save.disabled = True
        ranges = {ep: (tuple(span) if span is not None else None) for ep, span in spans.items()}

        def run() -> None:
            def progress(message: str) -> None:
                state["status"] = message
                dirty.set()
            try:
                path = trim_dataset(args.repo_id, args.root, output.value.strip(), args.output_root, ranges, progress)
                progress(f"Saved {path}")
                print(f"Saved {path}")
            except Exception as error:  # shown in the page; the source is never touched
                progress(f"Save failed: {error}")
                print(f"Save failed: {error}")
            finally:
                state["saving"] = False
                save.disabled = False

        threading.Thread(target=run, daemon=True).start()

    def render() -> None:
        ep, f = current(), int(frame.value)
        span = spans[ep]
        cut = span is None or not span[0] <= f <= span[1]
        if span is None:
            status = "EPISODE DROPPED"
        else:
            status = "CUT" if cut else "KEPT"
        kept = "dropped" if span is None else f"keep {span[0]}..{span[1]} ({span[1] - span[0] + 1} frames)"
        info = [f"episode {ep}  frame {f}/{lengths[ep] - 1}", status, f"t = {f / src.fps:.2f} s", kept]
        server.scene.set_background_image(_mosaic(list(camera_images(ep, f)), args.tile_width, info, cut),
                                          format="jpeg", jpeg_quality=85)
        rows = ["| episode | frames | keep |", "|---|---|---|"]
        for e, n in lengths.items():
            s = spans[e]
            rows.append(f"| {e} | {n} | {'dropped' if s is None else f'{s[0]}..{s[1]} ({s[1] - s[0] + 1})'} |")
        summary.content = "\n".join(rows + ([f"\n**{state['status']}**"] if state["status"] else []))

    sync_controls()
    # viser moves to the next free port when the requested one is taken.
    print(f"Trim UI at http://localhost:{server.get_port()}  (Ctrl+C to quit)")
    last_step = time.time()
    try:
        while True:
            if state["playing"]:
                time.sleep(max(0.0, 1.0 / (src.fps * float(speed.value)) - (time.time() - last_step)))
                last_step = time.time()
                ep, f = current(), int(frame.value)
                span = spans[ep]
                lo, hi = (span[0], span[1]) if (loop_kept.value and span is not None) else (0, lengths[ep] - 1)
                frame.value = lo if not lo <= f < hi else f + 1
                render()
            elif dirty.wait(0.05):
                dirty.clear()
                render()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


# ----------------------------------------------------------------------------------- CLI


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog="\n\n".join(__doc__.split("\n\n")[1:]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Source dataset, e.g. Sorozco0612/tblock_demos")
    parser.add_argument("--root", help="Source dataset directory (default: HF_LEROBOT_HOME/<repo-id>)")
    parser.add_argument("--output-repo-id", help="Trimmed copy's repo id (default: <repo-id>_trimmed)")
    parser.add_argument("--output-root", help="Trimmed copy's directory (default: HF_LEROBOT_HOME/<output-repo-id>)")
    parser.add_argument("--keep", nargs="+", metavar="EP:FIRST:LAST",
                        help="Keep frames FIRST..LAST (inclusive) of episode EP; skips the UI")
    parser.add_argument("--drop", nargs="+", type=int, metavar="EP", help="Drop whole episodes; skips the UI")
    parser.add_argument("--port", type=int, default=8080, help="viser port for the UI")
    parser.add_argument("--tile-width", type=int, default=640, help="Width of each camera tile in the UI (px)")
    args = parser.parse_args(argv)
    if args.keep or args.drop:
        ranges = parse_ranges(args.keep, args.drop)
        path = trim_dataset(args.repo_id, args.root, args.output_repo_id or f"{args.repo_id}_trimmed",
                            args.output_root, ranges)
        print(f"Saved {path}")
    else:
        trim_ui(args)


if __name__ == "__main__":
    main()
