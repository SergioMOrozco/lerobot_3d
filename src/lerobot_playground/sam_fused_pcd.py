# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
"""
Run SAM2 (+ Grounding DINO) segmentation, fuse masked depth from multiple cameras
into one point cloud per frame, optional geometry-only cleanup (no velocities),
write recording_dir/pcd_clean/*.npz (vels are zeros for file compatibility), and visualize.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

_PKG_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PKG_DIR.parent
_REPO_ROOT = str(_SRC_DIR.parent)

# Default camera folder names under recording_dir (must match calibration JSON keys)
DEFAULT_SERIALS = ["044322073544", "244622072067"]


def resolve_sam2_checkpoint(user_path: str) -> str:
    """Resolve relative checkpoint path; try repo root, src/, then cwd."""
    if os.path.isfile(user_path):
        return os.path.abspath(user_path)
    if os.path.isabs(user_path):
        return user_path
    for base in (_REPO_ROOT, str(_SRC_DIR), str(_PKG_DIR), os.getcwd()):
        candidate = os.path.join(base, user_path)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(_REPO_ROOT, user_path)


def select_bounding_box_by_click_matplotlib(
    image_pil: Image.Image, input_boxes: np.ndarray, labels: list
) -> tuple[np.ndarray, list, int]:
    image = np.array(image_pil)
    selected_idx: dict = {"value": None}

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Click a bounding box to select it, then close the window")

    for i, (box, label) in enumerate(zip(input_boxes, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), w, h, fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 5, 5),
            f"{i}: {label}",
            color="yellow",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=2),
        )

    def point_in_box(x, y, box):
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        containing = []
        for i, box in enumerate(input_boxes):
            if point_in_box(x, y, box):
                area = (box[2] - box[0]) * (box[3] - box[1])
                containing.append((area, i))
        if containing:
            containing.sort()
            selected_idx["value"] = containing[0][1]
            print(f"Selected box {selected_idx['value']}: {labels[selected_idx['value']]}")
            plt.close(fig)
        else:
            print("Click inside one of the boxes.")

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if selected_idx["value"] is None:
        raise RuntimeError("No bounding box selected.")
    idx = selected_idx["value"]
    return input_boxes[idx : idx + 1], [labels[idx]], idx


def mp4_to_numpy_list(path: str) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    frames = [np.array(f) for f in reader]
    reader.close()
    return frames


def mp4_to_pil_list(path: str) -> list[Image.Image]:
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(Image.fromarray(np.array(frame)))
    reader.close()
    return frames


def mask_frame_to_2d(mask_frame: np.ndarray) -> np.ndarray:
    """(H,W) or (H,W,C) -> (H,W) uint8 in {0,1}."""
    if mask_frame.ndim == 2:
        m = mask_frame
    else:
        m = mask_frame[..., 0]
    if m.max() > 1:
        m = (m > 127).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)
    return m


def depth2pcd(depth: np.ndarray, serial: str, mask: np.ndarray | None, calib_dir: str) -> np.ndarray:
    depth = np.ascontiguousarray(depth, dtype=np.float64)
    if mask is not None:
        m2 = mask_frame_to_2d(mask)
        depth = depth.copy()
        depth[m2 == 0] = 0.0

    with open(os.path.join(calib_dir, "extrinsic_calibration.json")) as f:
        extrinsics = json.load(f)
    with open(os.path.join(calib_dir, "intrinsic_calibration_848.json")) as f:
        intrinsics = json.load(f)

    fl_x = intrinsics[serial]["fl_x"]
    fl_y = intrinsics[serial]["fl_y"]
    cx = intrinsics[serial]["cx"]
    cy = intrinsics[serial]["cy"]
    w = intrinsics[serial]["w"]
    h = intrinsics[serial]["h"]
    pinhole = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

    depth_f32 = np.ascontiguousarray(depth.astype(np.float32))
    depth_image = o3d.geometry.Image(depth_f32)
    pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        pinhole,
        depth_trunc=1e9,
        stride=1,
        project_valid_depth_only=False,
        depth_scale=1.0,
    )
    pointcloud.transform(extrinsics[serial]["X_WC"])
    return np.asarray(pointcloud.points)


def fused_pcd_for_frame(
    recording_dir: str,
    serials: list[str],
    frame_id: int,
    rgb_cache: dict[str, list[np.ndarray]],
    mask_cache: dict[str, list[np.ndarray]],
    depth_cache: dict[str, np.ndarray],
    calib_dir: str,
) -> o3d.geometry.PointCloud:
    pts_list = []
    colors_list = []
    for serial in serials:
        rgb = rgb_cache[serial][frame_id]
        mask = mask_cache[serial][frame_id]
        depth = depth_cache[serial][frame_id].astype(np.float64) / 1000.0
        m2 = mask_frame_to_2d(mask)

        pts = depth2pcd(depth, serial, m2, calib_dir=calib_dir)
        pts = pts.reshape(depth.shape[0], depth.shape[1], 3)
        pts = pts[m2 > 0]
        colors = rgb[m2 > 0].astype(np.float64) / 255.0

        pts_list.append(pts)
        colors_list.append(colors)

    pts = np.concatenate(pts_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def fused_pts_colors_for_frame(
    serials: list[str],
    frame_id: int,
    rgb_cache: dict[str, list[np.ndarray]],
    mask_cache: dict[str, list[np.ndarray]],
    depth_cache: dict[str, np.ndarray],
    calib_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused points and RGB colors (0–1) in world frame; no velocities."""
    pts_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    for serial in serials:
        rgb = rgb_cache[serial][frame_id]
        mask = mask_cache[serial][frame_id]
        depth = depth_cache[serial][frame_id].astype(np.float64) / 1000.0
        m2 = mask_frame_to_2d(mask)

        pts = depth2pcd(depth, serial, m2, calib_dir=calib_dir)
        pts = pts.reshape(depth.shape[0], depth.shape[1], 3)
        pts = pts[m2 > 0]
        colors = rgb[m2 > 0].astype(np.float64) / 255.0
        pts_list.append(pts)
        colors_list.append(colors)

    return np.concatenate(pts_list, axis=0), np.concatenate(colors_list, axis=0)


def apply_geometry_cleanup(
    pts: np.ndarray,
    colors: np.ndarray,
    *,
    remove_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Statistical outlier removal, 10k cap, Z-axis radius filter (no velocity / no KNN)."""
    if pts.shape[0] == 0:
        return pts, colors

    finite = np.isfinite(pts).all(axis=1) & np.isfinite(colors).all(axis=1)
    pts, colors = pts[finite], colors[finite]
    if pts.shape[0] == 0:
        return pts, colors

    if remove_outliers:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        new_outlier = None
        rm_iter = 0
        while new_outlier is None or len(new_outlier.points) > 0:
            _, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors=25, std_ratio=2.0 + rm_iter * 0.5
            )
            new_pcd = pcd.select_by_index(inlier_idx)
            new_outlier = pcd.select_by_index(inlier_idx, invert=True)
            pcd = new_pcd
            rm_iter += 1

        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

    if pts.shape[0] > 10000:
        downsample_indices = torch.randperm(pts.shape[0], device="cpu")[:10000].numpy()
        pts = pts[downsample_indices]
        colors = colors[downsample_indices]

    if pts.shape[0] == 0:
        return pts, colors

    if remove_outliers:
        pts_z = pts.copy()
        pts_z[:, :2] = 0
        pcd_z = o3d.geometry.PointCloud()
        pcd_z.points = o3d.utility.Vector3dVector(pts_z)
        _, inlier_idx = pcd_z.remove_radius_outlier(nb_points=100, radius=0.02)
        pts = pts[inlier_idx]
        colors = colors[inlier_idx]

    finite = np.isfinite(pts).all(axis=1) & np.isfinite(colors).all(axis=1)
    return pts[finite], colors[finite]


def load_recording_caches(
    recording_dir: str, serials: list[str]
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
    dict[str, np.ndarray],
]:
    rgb_cache = {s: mp4_to_numpy_list(os.path.join(recording_dir, s, "rgb.mp4")) for s in serials}
    mask_cache = {s: mp4_to_numpy_list(os.path.join(recording_dir, s, "mask.mp4")) for s in serials}
    depth_cache = {s: np.load(os.path.join(recording_dir, s, "depth.npz"))["depth"] for s in serials}
    return rgb_cache, mask_cache, depth_cache


def numpy_to_o3d_pointcloud(pts: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def list_pcd_clean_frame_ids(pcd_dir: str) -> list[int]:
    if not os.path.isdir(pcd_dir):
        return []
    ids: list[int] = []
    for name in os.listdir(pcd_dir):
        if name.endswith(".npz") and name[:-4].isdigit():
            ids.append(int(name[:-4]))
    return sorted(ids)


def o3d_from_pcd_clean_npz(npz_path: str) -> o3d.geometry.PointCloud:
    with np.load(npz_path) as z:
        pts = np.asarray(z["pts"])
        colors = np.asarray(z["colors"])
    return numpy_to_o3d_pointcloud(pts, colors)


def export_pcd_clean(
    recording_dir: str,
    serials: list[str],
    calib_dir: str,
    n_frames_tail_trim: int = 0,
    remove_outliers: bool = True,
) -> None:
    rgb_cache, mask_cache, depth_cache = load_recording_caches(recording_dir, serials)

    for s in serials:
        print(
            f"[export_pcd_clean] lengths serial={s}: rgb={len(rgb_cache[s])} "
            f"mask={len(mask_cache[s])} depth={depth_cache[s].shape[0]}"
        )
    lengths = [len(rgb_cache[s]) for s in serials]
    lengths += [len(mask_cache[s]) for s in serials]
    lengths += [depth_cache[s].shape[0] for s in serials]
    n_full = min(lengths)
    n_frames = n_full - n_frames_tail_trim
    if n_frames <= 0:
        raise RuntimeError("Not enough aligned frames for pcd_clean export.")
    if n_full < max(len(rgb_cache[s]) for s in serials):
        print(
            "[export_pcd_clean] Warning: min(rgb, mask, depth) is below some rgb length; "
            "tail frames are dropped. Re-run SAM or fix recordings so streams match."
        )

    pcd_dir = os.path.join(recording_dir, "pcd_clean")
    if os.path.exists(pcd_dir):
        shutil.rmtree(pcd_dir)
    os.makedirs(pcd_dir)

    n_written = 0
    for frame_id in range(n_frames):
        pts, colors = fused_pts_colors_for_frame(
            serials,
            frame_id,
            rgb_cache,
            mask_cache,
            depth_cache,
            calib_dir,
        )
        if pts.shape[0] == 0:
            print(f"[export_pcd_clean] skip frame {frame_id}: no points after masking")
            continue
        pts, colors = apply_geometry_cleanup(pts, colors, remove_outliers=remove_outliers)
        if pts.shape[0] == 0:
            print(f"[export_pcd_clean] skip frame {frame_id}: no points after cleanup")
            continue
        vels = np.zeros((pts.shape[0], 3), dtype=np.float64)
        np.savez_compressed(
            os.path.join(pcd_dir, f"{frame_id}.npz"),
            pts=pts,
            colors=colors,
            vels=vels,
        )
        n_written += 1
    print(
        f"[export_pcd_clean] Wrote {n_written} npz file(s) under {pcd_dir} "
        f"(frame indices 0..{n_frames - 1}, tail_trim={n_frames_tail_trim}; "
        f"{n_frames - n_written} frame(s) skipped as empty)."
    )


def run_sam2_masks(
    recording_dir: str,
    text_prompts: str,
    serials: list[str],
    sam2_checkpoint: str,
    sam2_config: str,
    grounding_model_id: str,
    *,
    sam_video_chunk_frames: int = 50,
) -> None:
    image_predictor = SAM2ImagePredictor(build_sam2(sam2_config, sam2_checkpoint))
    video_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)

    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), amp:
        for serial in serials:
            r = os.path.join(recording_dir, serial, "rgb.mp4")
            rgb_frames_full_pil = mp4_to_pil_list(r)
            rgb_frames_full_np = mp4_to_numpy_list(r)
            n_frames = len(rgb_frames_full_pil)
            mask_video: list[np.ndarray] = []
            chunk_stride = (
                sam_video_chunk_frames if sam_video_chunk_frames > 0 else n_frames
            )

            for pivot_frame in range(0, n_frames, chunk_stride):
                chunk_end = min(pivot_frame + chunk_stride, n_frames)
                n_chunk_out = chunk_end - pivot_frame
                masks = np.zeros((1, 1))
                ann_frame = pivot_frame + 1
                objects: list = []
                input_boxes = np.zeros((0, 4))

                while masks.sum() == 0:
                    if ann_frame < 0:
                        raise RuntimeError("[run_sam2_masks] Could not find a frame with a valid mask.")
                    ann_frame -= 1
                    print(f"[run_sam2_masks] serial={serial} pivot={pivot_frame} try ann_frame={ann_frame}")

                    image = rgb_frames_full_pil[ann_frame]
                    inputs = processor(images=image, text=text_prompts, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = grounding_model(**inputs)
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        threshold=0.25,
                        target_sizes=[image.size[::-1]],
                    )
                    input_boxes = results[0]["boxes"].cpu().numpy()
                    objects = results[0]["labels"]

                    if len(objects) == 0:
                        continue

                    if len(objects) > 1:
                        input_boxes, objects, _ = select_bounding_box_by_click_matplotlib(
                            image_pil=image,
                            input_boxes=input_boxes,
                            labels=objects,
                        )

                    image_predictor.set_image(np.array(image))
                    masks, scores, logits = image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
                    if masks.ndim == 3:
                        masks = masks[None]
                    elif masks.ndim == 4:
                        raise ValueError("Unexpected mask rank from SAM2.")

                # Include every global frame in [pivot_frame, chunk_end): tmp must start at
                # min(ann_frame, pivot_frame) so we never drop frames when ann_frame > pivot.
                tmp_start = min(ann_frame, pivot_frame)
                ann_frame_idx = ann_frame - tmp_start
                rgb_frames_chunk_np = rgb_frames_full_np[tmp_start:chunk_end]
                tmp_len = len(rgb_frames_chunk_np)
                tmp_path = os.path.join(recording_dir, serial, "tmp.mp4")
                imageio.mimsave(tmp_path, rgb_frames_chunk_np, fps=30, codec="libx264")

                inference_state = video_predictor.init_state(video_path=tmp_path)
                for object_id, (_, box) in enumerate(zip(objects, input_boxes), start=1):
                    video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=object_id,
                        box=box,
                    )

                video_segments: dict = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                    inference_state
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                del inference_state
                torch.cuda.empty_cache()

                out_start = pivot_frame - tmp_start
                out_end = chunk_end - tmp_start
                if out_start < 0 or out_end > tmp_len:
                    raise RuntimeError(
                        f"[run_sam2_masks] serial={serial} pivot={pivot_frame} chunk_end={chunk_end} "
                        f"ann_frame={ann_frame} tmp_start={tmp_start} tmp_len={tmp_len} "
                        f"out_range=[{out_start},{out_end})"
                    )
                for ti in range(out_start, out_end):
                    if ti not in video_segments:
                        raise RuntimeError(
                            f"[run_sam2_masks] serial={serial} missing SAM output for tmp frame {ti} "
                            f"(expected 0..{tmp_len - 1}, got {len(video_segments)} keys)."
                        )
                    segments = video_segments[ti]
                    mask_list = list(segments.values())
                    masks_arr = np.concatenate(mask_list, axis=0)
                    if masks_arr.shape[0] > 1:
                        merged = np.logical_or.reduce(masks_arr, axis=0, keepdims=True)
                    else:
                        merged = masks_arr
                    mask_video.append(merged[0])

            if len(mask_video) != n_frames:
                raise RuntimeError(
                    f"[run_sam2_masks] serial={serial}: mask has {len(mask_video)} frames "
                    f"but rgb.mp4 has {n_frames}; alignment bug or SAM/video decode mismatch."
                )

            out_mask = os.path.join(recording_dir, serial, "mask.mp4")
            imageio.mimsave(
                out_mask,
                (np.array(mask_video).astype(np.uint8) * 255),
                fps=30,
                codec="libx264",
            )
            print(f"[run_sam2_masks] Wrote {out_mask} ({len(mask_video)} frames)")


def visualize_fused(
    recording_dir: str,
    serials: list[str],
    calib_dir: str,
    start_frame: int = 0,
    play_fps: float = 0.0,
    *,
    use_cleaned: bool = True,
    remove_outliers: bool = True,
    n_frames_tail_trim: int = 0,
    prefer_pcd_clean_on_disk: bool = True,
) -> None:
    pcd_dir = os.path.join(recording_dir, "pcd_clean")
    disk_frame_ids = list_pcd_clean_frame_ids(pcd_dir)
    use_disk = (
        use_cleaned
        and prefer_pcd_clean_on_disk
        and len(disk_frame_ids) > 0
    )

    rgb_cache = mask_cache = depth_cache = None
    recompute_cleaned = False

    if use_disk:
        print(
            f"[visualize_fused] Loading {len(disk_frame_ids)} frames from {pcd_dir} (fast path)."
        )
        n_frames = len(disk_frame_ids)
        mode_label = "cleaned (disk)"
    elif use_cleaned:
        rgb_cache, mask_cache, depth_cache = load_recording_caches(recording_dir, serials)
        lengths = [len(rgb_cache[s]) for s in serials]
        lengths += [len(mask_cache[s]) for s in serials]
        lengths += [depth_cache[s].shape[0] for s in serials]
        n_full = min(lengths)
        n_frames = n_full - n_frames_tail_trim
        if n_frames <= 0:
            raise RuntimeError("Not enough frames for cleaned visualization (check tail trim).")
        recompute_cleaned = True
        mode_label = "cleaned (recompute)"
    else:
        rgb_cache, mask_cache, depth_cache = load_recording_caches(recording_dir, serials)
        lengths = [len(rgb_cache[s]) for s in serials]
        lengths += [len(mask_cache[s]) for s in serials]
        lengths += [depth_cache[s].shape[0] for s in serials]
        n_frames = min(lengths)
        if n_frames <= 0:
            raise RuntimeError("No frames found (check rgb/mask/depth lengths).")
        mode_label = "raw fused"

    frame_idx = max(0, min(start_frame, n_frames - 1))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    oo = "on" if remove_outliers else "off"
    title_extra = "" if use_disk else f", outlier rm {oo}"
    vis.create_window(
        window_name=f"Fused PCD ({mode_label}{title_extra}) | Space/n next | p prev | q quit"
        + (f" | {play_fps} fps" if play_fps > 0 else ""),
        width=1280,
        height=720,
    )

    def build_pcd(slot_idx: int) -> o3d.geometry.PointCloud:
        if use_disk:
            fid = disk_frame_ids[slot_idx]
            path = os.path.join(pcd_dir, f"{fid}.npz")
            return o3d_from_pcd_clean_npz(path)
        if recompute_cleaned and rgb_cache is not None:
            pts, colors = fused_pts_colors_for_frame(
                serials, slot_idx, rgb_cache, mask_cache, depth_cache, calib_dir
            )
            if pts.shape[0] == 0:
                return numpy_to_o3d_pointcloud(pts, colors)
            pts, colors = apply_geometry_cleanup(pts, colors, remove_outliers=remove_outliers)
            return numpy_to_o3d_pointcloud(pts, colors)
        assert rgb_cache is not None
        return fused_pcd_for_frame(
            recording_dir, serials, slot_idx, rgb_cache, mask_cache, depth_cache, calib_dir
        )

    pcd = build_pcd(frame_idx)
    vis.add_geometry(pcd)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
    vis.add_geometry(axis)

    def refresh():
        nonlocal pcd
        vis.remove_geometry(pcd, reset_bounding_box=False)
        pcd = build_pcd(frame_idx)
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        if use_disk:
            fid = disk_frame_ids[frame_idx]
            print(
                f"Frame {frame_idx} / {n_frames - 1}  (file {fid}.npz, {len(pcd.points)} pts, {mode_label})"
            )
        else:
            print(f"Frame {frame_idx} / {n_frames - 1}  ({len(pcd.points)} points, {mode_label})")

    def next_frame(vis_):
        nonlocal frame_idx
        frame_idx = (frame_idx + 1) % n_frames
        refresh()
        return False

    def prev_frame(vis_):
        nonlocal frame_idx
        frame_idx = (frame_idx - 1 + n_frames) % n_frames
        refresh()
        return False

    # GLFW uses different codes for n vs N; register both (and Space) so stepping works with caps off.
    for _key in (ord("n"), ord("N"), ord(" ")):
        vis.register_key_callback(_key, next_frame)
    for _key in (ord("p"), ord("P")):
        vis.register_key_callback(_key, prev_frame)

    if play_fps > 0:
        last_advance = time.monotonic()

        def anim(vis_):
            nonlocal frame_idx, last_advance
            now = time.monotonic()
            if now - last_advance >= 1.0 / play_fps:
                last_advance = now
                frame_idx = (frame_idx + 1) % n_frames
                refresh()
            return False

        vis.register_animation_callback(anim)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.08])
    refresh()

    print(
        f"Viewer ({mode_label}): Space or n = next frame, p = previous, q = quit."
        + (f" Auto-advance at {play_fps} fps." if play_fps > 0 else "")
    )
    vis.run()
    vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM segmentation + fused multi-camera point clouds")
    parser.add_argument("recording_dir", type=str, help="Directory containing per-serial rgb.mp4, depth.npz")
    parser.add_argument(
        "--text-prompts",
        type=str,
        default="cloth.",
        help="Grounding DINO text prompt (same as postprocess)",
    )
    parser.add_argument(
        "--serials",
        type=str,
        default=",".join(DEFAULT_SERIALS),
        help="Comma-separated serial folder names under recording_dir",
    )
    parser.add_argument(
        "--skip-sam",
        action="store_true",
        help="Do not run SAM; expect mask.mp4 under each serial folder",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=".",
        help="Directory with extrinsic_calibration.json and intrinsic_calibration_848.json",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default=os.environ.get(
            "SAM2_CHECKPOINT", "models/weights/sam2/sam2.1_hiera_large.pt"
        ),
        help="Path to sam2.1_hiera_large.pt (weights). Env SAM2_CHECKPOINT overrides default.",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Hydra config name inside the installed sam2 package (not a path under this repo)",
    )
    parser.add_argument(
        "--grounding-model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
    )
    parser.add_argument(
        "--sam-video-chunk",
        type=int,
        default=0,
        help="Max frames per SAM2 video propagation chunk (clips to each rgb.mp4 length). "
        "Use 0 for one chunk covering the full video (higher VRAM use).",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--play-fps",
        type=float,
        default=0.0,
        help="If > 0, advance frames automatically at this rate (no keypress needed).",
    )
    parser.add_argument(
        "--save-pcd-clean",
        action="store_true",
        help="Export fused clouds with geometry cleanup to recording_dir/pcd_clean/ (vels in npz are zeros).",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip Open3D viewer (useful with --save-pcd-clean only).",
    )
    parser.add_argument(
        "--pcd-tail-trim",
        type=int,
        default=0,
        help="Drop this many frames at the end after aligning rgb/mask/depth (default 0 = full overlap).",
    )
    parser.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Skip statistical + Z-axis radius outlier steps (still downsamples to 10k).",
    )
    parser.add_argument(
        "--vis-raw",
        action="store_true",
        help="Visualize raw fused clouds (no cleanup). Export unchanged.",
    )
    parser.add_argument(
        "--recompute-cleaned-vis",
        action="store_true",
        help="Do not read pcd_clean/*.npz; re-fuse from rgb/mask/depth (slow).",
    )
    args = parser.parse_args()

    serials = [s.strip() for s in args.serials.split(",") if s.strip()]
    recording_dir = os.path.abspath(args.recording_dir)
    calib_dir = os.path.abspath(args.calib_dir)
    remove_outliers = not args.no_outlier_removal

    ckpt = resolve_sam2_checkpoint(args.sam2_checkpoint)
    if not args.skip_sam and not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"SAM2 weights not found: {ckpt}\n"
            "The YAML under src/configs/... is only a Hydra recipe; you still need the checkpoint .pt "
            "(download SAM 2.1 Hiera Large from the official release).\n"
            "Then either:\n"
            f"  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt\n"
            "  or set SAM2_CHECKPOINT to that path.\n"
            f"Tried relative path {args.sam2_checkpoint!r} under: repo root, src/, package dir, cwd."
        )
    # SAM2 uses Hydra (pkg://sam2); config name lives inside the installed sam2 package, not src/configs/.
    cfg = args.sam2_config

    if not args.skip_sam:
        run_sam2_masks(
            recording_dir,
            args.text_prompts,
            serials,
            sam2_checkpoint=ckpt,
            sam2_config=cfg,
            grounding_model_id=args.grounding_model,
            sam_video_chunk_frames=args.sam_video_chunk,
        )

    if args.save_pcd_clean:
        export_pcd_clean(
            recording_dir,
            serials,
            calib_dir,
            n_frames_tail_trim=args.pcd_tail_trim,
            remove_outliers=remove_outliers,
        )

    if not args.no_vis:
        visualize_fused(
            recording_dir,
            serials,
            calib_dir,
            start_frame=args.start_frame,
            play_fps=args.play_fps,
            use_cleaned=not args.vis_raw,
            remove_outliers=remove_outliers,
            n_frames_tail_trim=args.pcd_tail_trim,
            prefer_pcd_clean_on_disk=not args.recompute_cleaned_vis,
        )


if __name__ == "__main__":
    main()
