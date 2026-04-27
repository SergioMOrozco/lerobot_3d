# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.
import os

import yaml
from typing import Union
from pathlib import Path
import argparse
import os
import subprocess
import numpy as np
import glob
import cv2
import torch
import shutil
import json
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw
import supervision as sv
import open3d as o3d
import time
import imageio
#from modules_teleop.perception import PerceptionModule

# cotracker
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer

# sam2
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


import matplotlib.pyplot as plt

import cv2
import numpy as np

def select_bounding_box_by_click_matplotlib(image_pil, input_boxes, labels):
    """
    Click a bounding box to select it.

    Returns:
        selected_box: shape (1, 4)
        selected_label: list[str] length 1
        selected_idx: int
    """
    image = np.array(image_pil)
    selected_idx = {"value": None}

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
    return input_boxes[idx:idx+1], [labels[idx]], idx


def select_bounding_box_by_click(image_pil, input_boxes, labels, window_name="Select Box"):
    """
    Let the user click one of the candidate bounding boxes.

    Args:
        image_pil: PIL image
        input_boxes: numpy array of shape (N, 4) in xyxy format
        labels: list of length N

    Returns:
        selected_box: numpy array of shape (1, 4)
        selected_label: list with one label
        selected_idx: int
    """
    image = np.array(image_pil).copy()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    selected_idx = {"value": None}
    click_pt = {"value": None}

    def draw_boxes(img, highlight_idx=None):
        disp = img.copy()
        for i, (box, label) in enumerate(zip(input_boxes, labels)):
            x1, y1, x2, y2 = map(int, box.tolist())

            color = (0, 255, 0) if i == highlight_idx else (0, 0, 255)
            thickness = 3 if i == highlight_idx else 2

            cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
            text = f"{i}: {label}"
            cv2.putText(
                disp,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        return disp

    def point_in_box(x, y, box):
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt["value"] = (x, y)

            # First prefer boxes containing the click
            containing = []
            for i, box in enumerate(input_boxes):
                if point_in_box(x, y, box):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    containing.append((area, i))

            if len(containing) > 0:
                # choose smallest containing box
                containing.sort()
                selected_idx["value"] = containing[0][1]
            else:
                # if click is outside all boxes, choose nearest box center
                dists = []
                for i, box in enumerate(input_boxes):
                    cx = 0.5 * (box[0] + box[2])
                    cy = 0.5 * (box[1] + box[3])
                    d = (cx - x) ** 2 + (cy - y) ** 2
                    dists.append((d, i))
                dists.sort()
                selected_idx["value"] = dists[0][1]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        disp = draw_boxes(image_bgr, selected_idx["value"])
        cv2.putText(
            disp,
            "Click a box. Press ENTER/SPACE to confirm, ESC to cancel.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 32]:  # Enter or Space
            if selected_idx["value"] is not None:
                break
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            raise RuntimeError("User cancelled bounding box selection.")

    cv2.destroyWindow(window_name)

    idx = selected_idx["value"]
    selected_box = input_boxes[idx:idx + 1]
    selected_label = [labels[idx]]

    return selected_box, selected_label, idx

def get_bounding_box():
    return np.array([[-0.01, 0.8], [-0.01, 0.8], [-0.01, 0.8]])  # the world frame robot workspace

def depth2pcd(depth, serial, mask = None):

    if True:
        assert "Need serial specific calibration."

    with open("extrinsic_calibration.json", "r") as f:
        extrinsics = json.load(f)

    with open("intrinsic_calibration_848.json", "r") as f:
        intrinsics = json.load(f)

    if mask is not None:
        depth[mask == 0] = 0.0

    fl_x = intrinsics[serial]['fl_x']
    fl_y = intrinsics[serial]['fl_y']
    cx = intrinsics[serial]['cx']
    cy = intrinsics[serial]['cy']
    w = intrinsics[serial]['w']
    h = intrinsics[serial]['h']

    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)

    depth = np.ascontiguousarray(depth.astype(np.float32))
    depth_image = o3d.geometry.Image(depth)

    pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        intrinsics,
        depth_trunc=1e9,  # no truncation
        stride=1,         # do not subsample
        project_valid_depth_only=False,
        depth_scale=1.0,
    )

    pointcloud.transform(extrinsics[serial]['X_WC'])

    return np.array(pointcloud.points)

def mp4_to_pil_list(path):
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frame = Image.fromarray(np.array(frame))  # ensure RGB PIL image
        frames.append(frame)
    reader.close()
    return frames

def mp4_to_numpy_list(path):
    reader = imageio.get_reader(path)
    frames = [np.array(f) for f in reader]   # already RGB
    reader.close()
    return frames

class PostProcessor:

    def __init__(self, 
            recording_dir,
            text_prompts='cloth.',
        ):

        self.H, self.W = 480, 848

        self.recording_dir = recording_dir
        self.text_prompts = text_prompts

        self.bbox = get_bounding_box()

    def run_sam2(self):
        checkpoint = "models/weights/sam2/sam2.1_hiera_large.pt"

        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 

        image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


        serials = ["044322073544", "244622072067"]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for serial in serials:

                r = os.path.join(self.recording_dir, serial, "rgb.mp4")

                rgb_frames_full_pil = mp4_to_pil_list(r)
                rgb_frames_full_np = mp4_to_numpy_list(r)

                n_frames = len(rgb_frames_full_pil)
                seq_len = 50

                mask_video = []

                for pivot_frame in range(0, n_frames, seq_len):

                    #if os.path.exists(save_dir / "mask" / f'episode_{episode_id:04d}_camera_{cam}' / f'pivot_frame_{pivot_frame:06d}'):
                    #    continue

                    masks = np.zeros((1, 1))
                    ann_frame = pivot_frame + 1
                    ann_frame_idx = 0
                    objects = [None, None]
                    no_objs = False
                    
                    while masks.sum() == 0:
                        if ann_frame == 0:
                            import ipdb; ipdb.set_trace()
                        ann_frame -= 1
                        print(f"[run_sam2] Finding a frame with mask for frame {ann_frame}")

                        image =  rgb_frames_full_pil[ann_frame]

                        # ground
                        inputs = processor(images=image, text=self.text_prompts, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = grounding_model(**inputs)
                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            threshold=0.25,
                            target_sizes=[image.size[::-1]]
                        )

                        input_boxes = results[0]["boxes"].cpu().numpy()
                        objects = results[0]["labels"]

                        ### Convert PIL image to draw-able image
                        #draw = ImageDraw.Draw(image)

                        #for box in input_boxes:
                        #    x1, y1, x2, y2 = box.tolist()
                        #    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                        #image.show()   # or save

                        if len(objects) == 0:
                            print("No objects found. Looking back..")
                            continue
                            #assert "No objects found."

                        #if len(objects) > 1:
                        #    assert "Too many objects found."
                        if len(objects) > 1:
                            print("[run_sam2] Multiple objects found. Please click the box you want.")
                            input_boxes, objects, selected_idx = select_bounding_box_by_click_matplotlib(
                                image_pil=image,
                                input_boxes=input_boxes,
                                labels=objects,
                            )
                            print(f"[run_sam2] Selected box {selected_idx}: {objects[0]}")
                        #if len(objects) > 1:
                        #    print("[run_sam2] Multiple objects found. Please click the box you want.")
                        #    input_boxes, objects, selected_idx = select_bounding_box_by_click(
                        #        image_pil=image,
                        #        input_boxes=input_boxes,
                        #        labels=objects,
                        #        window_name=f"Select Box - frame {ann_frame}"
                        #    )
                        #    print(f"[run_sam2] Selected box {selected_idx}: {objects[0]}")

                        #TODO: Rather than do this stuff, just query use which bounding box to choose.

                        image_predictor.set_image(np.array(image))

                        masks, scores, logits = image_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )
                        if masks.ndim == 3:
                            masks = masks[None]
                            scores = scores[None]
                            logits = logits[None]
                        elif masks.ndim == 4:
                            raise "Multiple objects detected."
                    
                    # save this as a tmp video file to propagate over a chunk using SAM2
                    rgb_frames_chunk_np = rgb_frames_full_np[ann_frame: pivot_frame + seq_len ]

                    imageio.mimsave(
                        os.path.join(self.recording_dir, serial, "tmp.mp4"),
                        rgb_frames_chunk_np,
                        fps=30,
                        codec="libx264"
                    )

                    inference_state = video_predictor.init_state(video_path=os.path.join(self.recording_dir, serial, "tmp.mp4"))

                    # Using box prompt
                    for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=object_id,
                            box=box,
                        )

                    video_segments = {}  # video_segments contains the per-frame segmentation results
                    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    
                    del inference_state
                    torch.cuda.empty_cache()
                    
                    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(objects, start=1)}

                    # in order not to duplicate frames, only take the last seq_len frames
                    for idx, (frame_idx, segments) in enumerate(list(video_segments.items())[-seq_len:]):
                        
                        object_ids = list(segments.keys())
                        masks = list(segments.values())
                        masks = np.concatenate(masks, axis=0)

                        if masks.shape[0] > 1:
                            assert multi_objs
                            masks_save = np.logical_or.reduce(masks, axis=0, keepdims=True)
                        else:
                            masks_save = masks

                        mask_video.append(masks_save[0])

                        #vis = True
                        #if vis:
                        #    img = rgb_frames_chunk_np[idx]
                        #    detections = sv.Detections(
                        #        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                        #        mask=masks, # (n, h, w)
                        #        class_id=np.array(object_ids, dtype=np.int32),
                        #    )
                        #    box_annotator = sv.BoxAnnotator()
                        #    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                        #    label_annotator = sv.LabelAnnotator()
                        #    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
                        #    mask_annotator = sv.MaskAnnotator()
                        #    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                        #    cv2.imwrite("debug.png", annotated_frame)
                        #    cv2.waitKey(100)

                imageio.mimsave(
                    os.path.join(self.recording_dir, serial, "mask.mp4"),
                    np.array(mask_video).astype(int) * 255,
                    fps=30,
                    codec="libx264"
                )

    def get_tracking(self):
            cotracker_predictor = CoTrackerPredictor(checkpoint="models/weights/cotracker/scaled_offline.pth", v2=False, offline=True, window_len=60).to('cuda')

            serials = ["044322073544", "244622072067"]

            for serial in serials:
                visualizer = Visualizer(save_dir=os.path.join(self.recording_dir, serial, "results_tracking"), pad_value=120, linewidth=3)

                r = os.path.join(self.recording_dir, serial, "rgb.mp4")
                m_r = os.path.join(self.recording_dir, serial, "mask.mp4")
                d_r = os.path.join(self.recording_dir, serial, "depth.npz")

                rgb_frames_full_np = mp4_to_numpy_list(r)
                mask_frames_full_np = (np.array(mp4_to_numpy_list(m_r))[:, :, :,0] / 255).astype(np.uint8)

                with np.load(d_r) as data:
                    depth_frames_full_np = data['depth']

                start_frame = 0

                n_frames = len(rgb_frames_full_np) 
                pivot_skip = 5
                seq_len = 15

                velocities = []

                # iterate 0, 5, 10, etc
                for pivot_frame in range(start_frame, n_frames, pivot_skip):  # determine the speed for frame (pivot_frame, pivot_frame + pivot_skip)

                    mask_pivot = mask_frames_full_np[pivot_frame]

                    img = rgb_frames_full_np[pivot_frame]

                    img_masked = img.copy() * (mask_pivot[:, :, None] > 0)

                    frames = []

                    # find the bounding box of the mask
                    no_mask = False
                    mask_min_h_all, mask_min_w_all, mask_max_h_all, mask_max_w_all = self.H, self.W, 0, 0

                    # pivot_frame, pivot_frame + 1, pivot_frame + 2, etc. up to pivot_frame + seq_len
                    for frame_id in range(pivot_frame, min(pivot_frame + seq_len, n_frames)):

                        mask = mask_frames_full_np[frame_id]

                        if mask.sum() == 0:
                            no_mask = True
                            break

                        # get bounding box of each frame
                        ys, xs = np.where(mask > 0)

                        pad = 5
                        mask_min_h = max(0, ys.min() - pad)
                        mask_max_h = min(mask.shape[0], ys.max() + pad)
                        mask_min_w = max(0, xs.min() - pad)
                        mask_max_w = min(mask.shape[1], xs.max() + pad)

                        mask_min_h_all = min(mask_min_h_all, mask_min_h)
                        mask_min_w_all = min(mask_min_w_all, mask_min_w)
                        mask_max_h_all = max(mask_max_h_all, mask_max_h)
                        mask_max_w_all = max(mask_max_w_all, mask_max_w)

                    if no_mask:
                        mask_min_h_all = 0
                        mask_max_h_all = 200
                        mask_min_w_all = 0
                        mask_max_w_all = 200
                    else:
                        # Make bounding box a square for some reason? I gues cotracker likes squares
                        center = ((mask_max_h_all + mask_min_h_all) // 2, (mask_max_w_all + mask_min_w_all) // 2)
                        max_w_h = max(mask_max_h_all - mask_min_h_all, mask_max_w_all - mask_min_w_all)
                        mask_min_h_all = max(0, center[0] - max_w_h // 2)
                        mask_max_h_all = min(self.H, center[0] + max_w_h // 2)
                        mask_min_w_all = max(0, center[1] - max_w_h // 2)
                        mask_max_w_all = min(self.W, center[1] + max_w_h // 2)
                    
                    # crop image and masks according to bounding box, upscale them and put them all into a list
                    for frame_id in range(pivot_frame, min(pivot_frame + seq_len, n_frames)):
                        
                        mask = mask_frames_full_np[frame_id]
                        img = rgb_frames_full_np[frame_id]
                        
                        # crop image and mask according to bounding box, then upscale them
                        img = img[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]
                        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_LINEAR)

                        mask = mask[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]
                        mask = cv2.resize(mask, (mask.shape[1] * 4, mask.shape[0] * 4), interpolation=cv2.INTER_NEAREST)

                        img_masked = img.copy() * (mask[:, :, None] > 0)
                        frames.append(torch.tensor(img_masked))
                    
                    video = torch.stack(frames).permute(0, 3, 1, 2)[None].float().to('cuda')  # B T C H W

                    # crop mask according to bounding box
                    mask_pivot = mask_pivot[mask_min_h_all:mask_max_h_all, mask_min_w_all:mask_max_w_all]

                    # shrink segmentation mask slightly to reduce 1 pixel edges, and producing a tighter and cleaner mask so cotracker only tracks points inside stable part of object
                    mask_pivot = cv2.erode(mask_pivot, np.ones((3, 3), np.uint8), iterations=1)

                    pred_tracks, pred_visibility = cotracker_predictor(
                        video, 
                        segm_mask=torch.from_numpy(mask_pivot).float().to('cuda')[None, None],
                        grid_size=80,
                    ) # B T N 2,  B T N
                    
                    # smooth the tracks multiple time (up to 3 times)
                    for _ in range(3):
                        for i in range(1, pred_tracks.shape[1] - 1):
                            #Weighted moving average so there is less jitter. More weight to current track (weight =2)
                            pred_tracks[:, i] = (2 * pred_tracks[:, i] + pred_tracks[:, i-1] + pred_tracks[:, i+1]) // 4
                    
                    vis = True
                    if vis:
                        visualizer.visualize(video, pred_tracks, pred_visibility, filename=f"tracks_{pivot_frame}")

                    # transform pred tracks and pred visibility to original image size
                    pred_tracks = pred_tracks.squeeze(0).cpu().numpy()  # T N 2
                    pred_visibility = pred_visibility.squeeze(0).cpu().numpy()  # T N 1
                    pred_tracks[:, :, 0] = pred_tracks[:, :, 0] / 4 + mask_min_w_all
                    pred_tracks[:, :, 1] = pred_tracks[:, :, 1] / 4 + mask_min_h_all
                    pred_tracks = pred_tracks[:, :, ::-1].copy()
                    
                    # calculate point speed in 3D
                    gap = 5
                    # Target frames starts at pivot frame
                    for target_frame in range(pivot_frame, min(pivot_frame + pivot_skip, n_frames - gap)):
                        depth_now = depth_frames_full_np[target_frame] / 1000.0
                        try:
                            depth_future = depth_frames_full_np[target_frame + gap] / 1000.0
                        except:
                            import ipdb; ipdb.set_trace()
                        
                        mask_now = mask_frames_full_np[target_frame]

                        # Gets all pixel coordinates the belong to object
                        mask_now_xy = np.where(mask_now > 0)
                        mask_now_xy = np.stack(mask_now_xy, axis=1)  # (n, 2)

                        # Gets all predicted tracker points that belong to object
                        track_now = pred_tracks[target_frame - pivot_frame]
                        vis_now = pred_visibility[target_frame - pivot_frame]

                        if len(track_now) == 0:
                            indices = np.zeros((0, 4)).astype(int)
                        else:
                            # each masked pixel is associated with the nearest track points
                            # indices.shape = (N_pixels, 4). Each pixel has 4 nearest tracker points
                            knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(track_now)
                            _, indices = knn.kneighbors(mask_now_xy)

                        mask_future = mask_frames_full_np[target_frame + gap]

                        # Gets all pixel coordinates that belong to object
                        mask_future_xy = np.where(mask_future > 0)
                        mask_future_xy = np.stack(mask_future_xy, axis=1)  # (n, 2)

                        # Gets all predicted tracker points that belong to object
                        track_future = pred_tracks[target_frame - pivot_frame + gap]
                        vis_future = pred_visibility[target_frame - pivot_frame + gap]

                        # for each pixel, get the x,y coordinate of each tracker in its knn
                        tracks_indices = track_now[indices]  # (n, k, 2)
                        tracks_future_indices = track_future[indices]  # (n, k, 2)

                        # for each pixel, get the visibility of each tracker in its knn
                        vis_indices = vis_now[indices]  # (n, k, 1)
                        vis_indices = np.all(vis_indices, axis=1).reshape(-1) # all neighbors must be 1, otherwise pixel gets a single 0 for its visibility
                        vis_future_indices = vis_future[indices]  # (n, k, 1)
                        vis_future_indices = np.all(vis_future_indices, axis=1).reshape(-1) # all neighbors must be 1, otherwise pixel gets a single 0 for its visibility

                        # for each pixel, get average position of track  from knn (belief state of where pixel is)
                        pred_mask_now_xy = np.round(tracks_indices.mean(axis=1)).astype(int)  # (n, 2)
                        pred_mask_now_xy[:, 0] = np.clip(pred_mask_now_xy[:, 0], 0, self.H - 1)
                        pred_mask_now_xy[:, 1] = np.clip(pred_mask_now_xy[:, 1], 0, self.W - 1)

                        # for each pixel, get average position of track  from knn (beliefe state of where pixel went)
                        pred_mask_future_xy = np.round(tracks_future_indices.mean(axis=1)).astype(int)  # (n, 2)  # actually we dont need to round here
                        pred_mask_future_xy[:, 0] = np.clip(pred_mask_future_xy[:, 0], 0, self.H - 1)
                        pred_mask_future_xy[:, 1] = np.clip(pred_mask_future_xy[:, 1], 0, self.W - 1)

                        ## extract depth and project to world coordinates
                        points_now = depth2pcd(depth_now, serial)
                        points_future = depth2pcd(depth_future, serial)

                        # make it into shape of image
                        points_now = points_now.reshape(depth_now.shape[0], depth_now.shape[1], 3)
                        points_future = points_future.reshape(depth_future.shape[0], depth_future.shape[1], 3)

                        # mask points
                        depth_threshold = [0.0, 2.0]

                        # build mask of valid pixels
                        depth_mask_now = np.logical_and((depth_now > depth_threshold[0]), (depth_now < depth_threshold[1]))  # (H, W)

                        # keep only points inside a 3d bounding box
                        depth_mask_now_bbox = np.logical_and(
                            np.logical_and(points_now[:, :, 0] > self.bbox[0][0], points_now[:, :, 0] < self.bbox[0][1]),
                            np.logical_and(points_now[:, :, 1] > self.bbox[1][0], points_now[:, :, 1] < self.bbox[1][1])
                        )  # does not include z axis

                        depth_mask_now_bbox = depth_mask_now_bbox.reshape(depth_now.shape[0], depth_now.shape[1])

                        # combine depth validity and workspace validity
                        depth_mask_now = np.logical_and(depth_mask_now, depth_mask_now_bbox)

                        #depth_mask_now_xy (N,) is just the N pixels with either a 1 or 0 for validity
                        depth_mask_now_xy = depth_mask_now[pred_mask_now_xy[:, 0], pred_mask_now_xy[:, 1]].reshape(-1)

                        # filter out any points that are not visible now or in the future
                        #TODO: Why would they want to mask out nonvisible tracks? This makes the tracking almost non-functional
                        #depth_mask_now_xy = np.logical_and(depth_mask_now_xy > 0, np.logical_and(vis_indices, vis_future_indices))  # filter out invalid points
                        depth_mask_now_xy = depth_mask_now_xy > 0

                        # valid pixel points that satisfy:
                        #   - below depth threshold
                        #   - inside bounding box 
                        #   - visible in the present and in the future 
                        valid_idx = np.where(depth_mask_now_xy > 0)[0]
                        
                        # only keep valid pixels
                        mask_now_xy = mask_now_xy[valid_idx]
                        pred_mask_now_xy = pred_mask_now_xy[valid_idx]
                        pred_mask_future_xy = pred_mask_future_xy[valid_idx]

                        # build mask of valid pixels
                        depth_mask_future = np.logical_and((depth_future > depth_threshold[0]), (depth_future < depth_threshold[1]))  # (H, W)

                        # keep only points inside a 3d bounding box
                        depth_mask_future_bbox = np.logical_and(
                            np.logical_and(points_future[:, :, 0] > self.bbox[0][0], points_future[:, :, 0] < self.bbox[0][1]),
                            np.logical_and(points_future[:, :, 1] > self.bbox[1][0], points_future[:, :, 1] < self.bbox[1][1])
                        )  # does not include z axis

                        depth_mask_future_bbox = depth_mask_future_bbox.reshape(depth_future.shape[0], depth_future.shape[1])

                        # combine depth validity and workspace validity
                        depth_mask_future = np.logical_and(depth_mask_future, depth_mask_future_bbox)

                        #depth_mask_future_xy (N,) is just the N pixels with either a 1 or 0 for validity
                        depth_mask_future_xy = depth_mask_future[mask_future_xy[:, 0], mask_future_xy[:, 1]].reshape(-1)

                        # valid pixel points that satisfy:
                        #   - below depth threshold
                        #   - inside bounding box 
                        valid_idx_future = np.where(depth_mask_future_xy > 0)[0]

                        # only keep valid pixels
                        mask_future_xy_valid = mask_future_xy[valid_idx_future]

                        if valid_idx_future.shape[0] < 4 or valid_idx.shape[0] < 4:
                            print(f"Warning: not enough valid points for frame {target_frame}")
                            # if too few points, zero velocity
                            speed_now = np.zeros((self.H, self.W, 3))
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)

                        else:
                            # proj tracks
                            k = 4

                            # for each future predicted track location, we get the 4 nearest pixels in the future mask (get the 4 nearest pixels where we think the tracked partilce went)
                            knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(mask_future_xy_valid)
                            _, indices = knn.kneighbors(pred_mask_future_xy)

                            # produce list of pixel coordinates where each tracked point might have moved in the future
                            proj_mask_future_xy = mask_future_xy_valid[indices]  # (n, k, 2)
                            proj_mask_future_xy = proj_mask_future_xy.reshape(-1, 2)
                            proj_mask_future_xy = np.round(proj_mask_future_xy).astype(int)
                            proj_mask_future_xy[:, 0] = np.clip(proj_mask_future_xy[:, 0], 0, self.H - 1)
                            proj_mask_future_xy[:, 1] = np.clip(proj_mask_future_xy[:, 1], 0, self.W - 1)
                            
                            # pred_mask_now_xy: belief state where pixel is
                            # points_now: current points in 3d space from depth
                            points_now = points_now[pred_mask_now_xy[:, 0], pred_mask_now_xy[:, 1]].reshape(-1, 3)

                            # proj_mask_future_xy: belief state of where tracked point moved to (not just tracked particle, but average position of 4 nearest particles)
                            # points_future: future points in 3d space from depth
                            points_future = points_future[proj_mask_future_xy[:, 0], proj_mask_future_xy[:, 1]]

                            #TODO: get average position from above knn pixel locations
                            points_future = points_future.reshape(-1, k, 3).mean(axis=1)  # average the k points

                            speed = points_future - points_now  # actually velocity
                            speed /= (1. / 30. * gap)  # divide by the time interval
                            
                            #mask_now_xy: all pixels that belong to object
                            speed_now = np.zeros((self.H, self.W, 3))
                            speed_now[mask_now_xy[:, 0], mask_now_xy[:, 1]] = speed
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)
                            
                            # if velocity magnitude is way larger than typical values, it is an outlier
                            outlier_mask = speed_now_norm > min(1, speed_now_norm.mean() + 50 * speed_now_norm.std())
                            outlier_xy = np.stack(np.where(outlier_mask), axis=1)  # (n, 2)

                            # replace outlier with average of its 8 neighbors
                            for xy in outlier_xy:
                                speed_now[xy[0], xy[1]] = (speed_now[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2].sum(axis=(0, 1)) - speed_now[xy[0], xy[1]]) / 8
                            speed_now_norm = np.linalg.norm(speed_now, axis=2)

                        # visualize the speed
                        #viz = True
                        #if viz:
                        #    speed_running_max = 0.5
                        #    _ = cv2.applyColorMap((speed_now_norm / speed_running_max * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        #    cv2.imwrite(save_dir_speed_cam / f"{target_frame:06d}.jpg", speed_now_norm / speed_running_max * 255)

                        # save the 3d vel
                        #np.savez_compressed(episode_data_dir_cam / "vel" / f"{target_frame:06d}.npz", vel=speed_now.astype(np.float16))

                        # we calculated the speed for the current timestep.
                        velocities.append(speed_now)

                        # vis depth mask
                        #depth_mask_vis = np.logical_and(depth_mask_now, mask_now)
                        #cv2.imwrite(episode_data_dir_cam / "depth_mask" / f"{target_frame:06d}.png", depth_mask_vis * 255)

                np.savez_compressed(os.path.join(self.recording_dir, serial, "velocities.npz"), velocities=np.array(velocities))

    def get_pcd(self):

        serials = ["044322073544", "244622072067"]

        rgb_frame_full_np_cache = {}
        mask_frame_full_np_cache = {}
        depth_frame_full_np_cache = {}
        vel_frame_full_np_cache = {}

        for serial in serials:
            rgb_frame_full_np_cache[serial] = mp4_to_numpy_list(os.path.join(self.recording_dir, serial, "rgb.mp4"))
            mask_frame_full_np_cache[serial] = (np.array(mp4_to_numpy_list(os.path.join(self.recording_dir, serial, "mask.mp4")))[:, :, :,0] / 255).astype(np.uint8)
            depth_frame_full_np_cache[serial] = np.load(os.path.join(self.recording_dir, serial, "depth.npz"))['depth']
            vel_frame_full_np_cache[serial] = np.load(os.path.join(self.recording_dir, serial, "velocities.npz"))['velocities']

        rgb_frames_full_np = mp4_to_numpy_list(os.path.join(self.recording_dir, serials[0], "rgb.mp4"))
        n_frames = len(rgb_frames_full_np) - 5
        #n_frames = 70

        pcd_dir = os.path.join(self.recording_dir, "pcd_clean")

        if os.path.exists(pcd_dir):
            shutil.rmtree(pcd_dir)

        os.makedirs(pcd_dir)

        for frame_id in range(n_frames):

            pts_list = []
            colors_list = []
            vels_list = []
            camera_indices_list = []

            for serial in serials:

                rgb_frames_full_np = rgb_frame_full_np_cache[serial]
                mask_frames_full_np = mask_frame_full_np_cache[serial]
                depth_frames_full_np = depth_frame_full_np_cache[serial]
                vel_frames_full_np = vel_frame_full_np_cache[serial]

                mask = mask_frames_full_np[frame_id]
                img = rgb_frames_full_np[frame_id]
                depth = depth_frames_full_np[frame_id] / 1000.0
                vel = vel_frames_full_np[frame_id]

                points = depth2pcd(depth, serial, mask=mask)
                points = points.reshape(depth.shape[0], depth.shape[1], 3)
                points = points[mask > 0]

                colors = img[mask > 0]
                vel = vel[mask > 0]

                assert points.shape[0] == vel.shape[0]
                #camera_indices = np.ones(points.shape[0]) * cam

                pts_list.append(points)
                colors_list.append(colors)
                vels_list.append(vel)
                #camera_indices_list.append(camera_indices)
                
            pts = np.concatenate(pts_list, axis=0)
            colors = np.concatenate(colors_list, axis=0)
            vels = np.concatenate(vels_list, axis=0)
            #camera_indices = np.concatenate(camera_indices_list)

            rm_outlier = True
            if rm_outlier:
                #camera_indices = camera_indices[:, None].repeat(9, axis=-1).reshape(pts.shape[0], 3, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(colors / 255)
                pcd.normals = o3d.utility.Vector3dVector(vels)  # fake normals. just a means to store velocity
                #pcd.covariances = o3d.utility.Matrix3dVector(camera_indices) # just a means of storing camera index inside particle

                outliers = None
                new_outlier = None
                rm_iter = 0

                # iteratively remove outliers
                while new_outlier is None or len(new_outlier.points) > 0:

                    # Find each particles nearest 25 neighbors.
                    # if the mean distance is greater than the global mean plus some extra, we mark it as an outlier
                    # inlier idx is just particles that are not outliers
                    _, inlier_idx = pcd.remove_statistical_outlier(
                        nb_neighbors = 25, std_ratio = 2.0 + rm_iter * 0.5
                    )
                    new_pcd = pcd.select_by_index(inlier_idx)
                    new_outlier = pcd.select_by_index(inlier_idx, invert=True)
                    if outliers is None:
                        outliers = new_outlier
                    else:
                        outliers += new_outlier
                    pcd = new_pcd
                    rm_iter += 1
                
                pts = np.array(pcd.points)
                colors = np.array(pcd.colors)
                vels = np.array(pcd.normals)
                #camera_indices = np.array(pcd.covariances)[:, 0, 0]

            ### downsample point cloud to 10000
            if pts.shape[0] > 10000:
                downsample_indices = torch.randperm(pts.shape[0])[: 10000]
                pts = pts[downsample_indices]
                colors = colors[downsample_indices]
                vels = vels[downsample_indices]
                #camera_indices = camera_indices[downsample_indices]

            n_pts_orig = pts.shape[0]

            # remove outliers based on height
            pts_z = pts.copy()
            pts_z[:, :2] = 0  # only consider z axis
            pcd_z = o3d.geometry.PointCloud()
            pcd_z.points = o3d.utility.Vector3dVector(pts_z)
            _, inlier_idx = pcd_z.remove_radius_outlier(
                nb_points = 100, radius = 0.02
            )
            pts = pts[inlier_idx]
            colors = colors[inlier_idx]
            vels = vels[inlier_idx]
            #camera_indices = camera_indices[inlier_idx]

            # remove outliers based on vel
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vels)  # fake points
            _, inlier_idx = pcd.remove_radius_outlier(
                nb_points = 20, radius = 0.01
            )
            pts = pts[inlier_idx]
            colors = colors[inlier_idx]
            vels = vels[inlier_idx]
            #camera_indices = camera_indices[inlier_idx]

            n_pts_clean = pts.shape[0]

            knn = NearestNeighbors(n_neighbors=20, algorithm='kd_tree').fit(pts)
            _, indices = knn.kneighbors(pts) # indices has shape (N,19)
            indices = indices[:, 1:]  # exclude the point itself
            dists = np.linalg.norm(pts[indices] - pts[:, None], axis=2) # subtract each point from its 19 neighbors (N,19,3) - (N,1,3) = (N,19)
            
            # convert distances to weights using exponential decay kernel
            # close neighbors, high weight
            # far neighbors, low weight
            weights = np.exp(-dists / 0.01)
            weights = weights / weights.sum(axis=1, keepdims=True) # normalize weights
            vels_smooth = (weights[:, :, None] * vels[indices]).sum(axis=1)
            vels = vels_smooth

            np.savez_compressed(os.path.join(self.recording_dir, "pcd_clean", f"{frame_id}.npz"), pts=pts, colors=colors, vels=vels)

    def vis_pcd(self):

        pcd_dir = os.path.join(self.recording_dir, "pcd_clean/")
        n_frames = 70

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()

        pcd_path = os.path.join(pcd_dir, f"0.npz")

        # geometry is the point cloud used in your animaiton
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(np.load(pcd_path)['pts'])
        geometry.colors = o3d.utility.Vector3dVector(np.load(pcd_path)['colors'])
        visualizer.add_geometry(geometry)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        visualizer.add_geometry(axis)
        line_set = o3d.geometry.LineSet()
        visualizer.add_geometry(line_set)

        # add a tabletop mesh
        #mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.01)
        #mesh = mesh.translate([0, -0.5, 0])
        #mesh.compute_vertex_normals()
        #mesh.paint_uniform_color([0.8, 0.8, 0.8])
        #visualizer.add_geometry(mesh)

        for i in range (1,71):
            for j in range(10):
                pcd_path = os.path.join(pcd_dir, f"{i}.npz")

                # now modify the points of your geometry
                # you can use whatever method suits you best, this is just an example
                pcd = np.load(pcd_path)
                points = pcd['pts']
                colors = pcd['colors']
                vels = pcd['vels']

                geometry.points = o3d.utility.Vector3dVector(points)
                geometry.colors = o3d.utility.Vector3dVector(colors)

                arrow_length = np.linalg.norm(vels, axis=1, keepdims=True)  # Length of each arrow

                # Create endpoint of each arrow by shifting each point in a given direction
                # Here we assume the direction for each arrow is a unit vector in (0, 0, 1) (upward).
                directions = vels / np.linalg.norm(vels, axis=1, keepdims=True)
                end_points = points + arrow_length * directions

                all_points = np.vstack([points, end_points])

                # Create a LineSet to represent arrows
                lines = [[i, i + len(points)] for i in range(len(points))]
                colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for arrows

                # Define the LineSet object
                line_set.points = o3d.utility.Vector3dVector(all_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                visualizer.update_geometry(geometry)
                #visualizer.update_geometry(line_set)
                visualizer.poll_events()
                visualizer.update_renderer()
                print(f"[vis_pcd] Frame {i} done")

    def vis_traj(self):
        pcd_dir = os.path.join(self.recording_dir, "pcd_clean")
        n_frames = 70
        #n_frames = 20

        start_frame = 0
        pivot_skip = 70
        seq_len = 70

        for pivot_frame in range(start_frame, n_frames, pivot_skip):
            pcd = np.load(os.path.join(pcd_dir, f"{pivot_frame}.npz"))
            points_0 = pcd['pts']
            colors_0 = pcd['colors']
            vels_0 = pcd['vels']

            points_list = [points_0]
            vels_list = [vels_0]

            gap = 1
            dt = 1. / 30 * gap
            for frame_id in range(pivot_frame + 1, min(pivot_frame + seq_len, n_frames), gap):
                pcd = np.load(os.path.join(pcd_dir, f"{frame_id}.npz"))
                points = pcd['pts']
                vels = pcd['vels']

                points_pred = points_list[-1] + vels_list[-1] * dt

                # knn
                knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(points)
                _, indices = knn.kneighbors(points_pred)

                vels_next = vels[indices].mean(axis=1)

                points_list.append(points_pred)
                vels_list.append(vels_next)
            
            points_list = np.stack(points_list, axis=0)
            vels_list = np.stack(vels_list, axis=0)

            vis = True
            if vis:
                visualizer = o3d.visualization.Visualizer()
                visualizer.create_window()

                # geometry is the point cloud used in your animaiton
                geometry = o3d.geometry.PointCloud()
                geometry.points = o3d.utility.Vector3dVector(points_0)
                geometry.colors = o3d.utility.Vector3dVector(colors_0)
                visualizer.add_geometry(geometry)
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                visualizer.add_geometry(axis)

                ## add a tabletop mesh
                #mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.01)
                #mesh = mesh.translate([0, -0.5, 0])
                #mesh.compute_vertex_normals()
                #mesh.paint_uniform_color([0.8, 0.8, 0.8])
                #visualizer.add_geometry(mesh)

                for i in range(len(points_list)):
                    for j in range(10):
                        # now modify the points of your geometry
                        # you can use whatever method suits you best, this is just an example
                        points = points_list[i]
                        colors = colors_0
                        vels = vels_list[i]

                        geometry.points = o3d.utility.Vector3dVector(points)
                        geometry.colors = o3d.utility.Vector3dVector(colors)

                        arrow_length = np.linalg.norm(vels, axis=1, keepdims=True)  # Length of each arrow

                        # Create endpoint of each arrow by shifting each point in a given direction
                        # Here we assume the direction for each arrow is a unit vector in (0, 0, 1) (upward).
                        directions = vels / np.linalg.norm(vels, axis=1, keepdims=True)
                        end_points = points + arrow_length * directions

                        all_points = np.vstack([points, end_points])

                        # Create a LineSet to represent arrows
                        lines = [[i, i + len(points)] for i in range(len(points))]
                        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for arrows

                        visualizer.update_geometry(geometry)
                        visualizer.poll_events()
                        visualizer.update_renderer()
                        print(f"[pred_traj] Frame {i} done")
            
                visualizer.destroy_window()
                input("Press Enter to continue...")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_prompts', type=str, default='')
    args = parser.parse_args()

    recording_dir = "recordings/rope_lift"

    pp = PostProcessor(recording_dir, args.text_prompts)
    #pp.run_sam2()
    #pp.get_tracking()
    pp.get_pcd()
    pp.vis_pcd()
    #pp.vis_traj()
    #pp.get_sub_episodes()
