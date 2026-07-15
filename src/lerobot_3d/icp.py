import open3d as o3d
from PIL import Image
import numpy as np
import json
import os
from lerobot_3d.icp_new import discover_calibration_serials
from lerobot_3d.point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from lerobot_3d.point_clouds.point_cloud_viewer import LivePointCloudViewer

def depth2pcd(depth, serial, color = None, T_wc= None, mask = None):

    with open("intrinsic_calibration.json", "r") as f:
        intrinsics = json.load(f)

    if mask is not None:
        depth = depth.copy()
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

    if color is not None:

        if color.dtype == np.float32:
            img_uint8 = np.array(color * 255, dtype=np.uint8)
        else:
            img_uint8 = np.array(color)

        color_image = o3d.geometry.Image(img_uint8)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics,
        )

    else:
        pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsics,
            depth_scale=1.0,
        )

    if T_wc is not None:
        pointcloud.transform(T_wc)

    return pointcloud


def _world_centroid(T, pts_cam):
    return (T[:3, :3] @ pts_cam.T).T.mean(axis=0) + T[:3, 3]


def _rotate_about_centroid(T_current, pts_cam, axis, angle_deg):
    """Rotate T_current about the world-frame centroid of pts_cam transformed by T_current."""
    centroid = _world_centroid(T_current, pts_cam)
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    Rm = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.deg2rad(angle_deg))
    T_rot = np.eye(4)
    T_rot[:3, :3] = Rm
    T_rot[:3, 3] = centroid - Rm @ centroid
    return T_rot @ T_current


def _translate_world(T_current, delta):
    T_t = np.eye(4)
    T_t[:3, 3] = delta
    return T_t @ T_current


def _pose_delta_summary(T_current, T_init, pts_cam):
    """Centroid displacement (cm) and rotation angle (deg) of T_current relative to T_init.

    Uses centroid displacement rather than the raw matrix translation column so a
    pure in-place rotation (pivoting off-origin) correctly reports ~0cm of "movement".
    """
    trans_cm = np.linalg.norm(
        _world_centroid(T_current, pts_cam) - _world_centroid(T_init, pts_cam)
    ) * 100.0
    T_delta = T_current @ np.linalg.inv(T_init)
    trace = np.clip((np.trace(T_delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = np.degrees(np.arccos(trace))
    return trans_cm, rot_deg


def manual_align_transform(
    pcd_cam,
    target_pcd,
    T_init,
    T_fallback=None,
    window_title="Manual align",
    translate_steps=(0.002, 0.01, 0.05),  # meters: fine / medium / coarse
    rotate_steps_deg=(1.0, 5.0, 20.0),
):
    """Interactive world-frame nudge of pcd_cam (via T_init) onto target_pcd.

    pcd_cam is the masked robot cloud in camera frame (untransformed); target_pcd
    (e.g. the mesh cloud) is displayed statically in world frame. Rotations pivot
    about pcd_cam's current world-frame centroid, so the cloud spins in place
    regardless of how far it's already been dragged.

    Displays starting from T_init. Escape (abort) reverts to T_fallback (defaults
    to T_init) -- pass e.g. a pre-ICP pose here when confirming an ICP result, so
    a bad ICP jump (nearest-neighbor distance alone can't tell "correct surface"
    from "close to the wrong nearby link/part", a real risk on self-proximate
    geometry like a robot arm) can be visually rejected and reverted, not just
    caught by a distance heuristic.

    Returns the confirmed 4x4 transform, or T_fallback if aborted.
    """
    if T_fallback is None:
        T_fallback = T_init
    pts_cam = np.asarray(pcd_cam.points)

    state = {"T": T_init.copy(), "step_idx": 0, "confirmed": False}

    def current_display_pcd():
        p = o3d.geometry.PointCloud(pcd_cam)
        p.transform(state["T"])
        return p

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{window_title} | arrows=XY, PgUp/PgDn=Z, 1-6=rotate XYZ, "
        "f=step size, r=reset, Enter=confirm, Esc=abort/revert",
        width=1280,
        height=720,
    )
    vis.add_geometry(target_pcd)
    moving = current_display_pcd()
    vis.add_geometry(moving)

    def report():
        trans_cm, rot_deg = _pose_delta_summary(state["T"], T_init, pts_cam)
        t_step = translate_steps[state["step_idx"]]
        r_step = rotate_steps_deg[state["step_idx"]]
        print(
            f"  step: translate={t_step * 100:.1f}cm rotate={r_step:.1f}deg  |  "
            f"delta from initial guess: {trans_cm:.2f}cm, {rot_deg:.2f}deg"
        )

    def refresh():
        nonlocal moving
        vis.remove_geometry(moving, reset_bounding_box=False)
        moving = current_display_pcd()
        vis.add_geometry(moving, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()

    def make_translate(axis_vec):
        def _cb(vis_):
            step = translate_steps[state["step_idx"]]
            state["T"] = _translate_world(state["T"], np.array(axis_vec) * step)
            refresh()
            report()
            return False
        return _cb

    def make_rotate(axis_vec):
        def _cb(vis_):
            step = rotate_steps_deg[state["step_idx"]]
            state["T"] = _rotate_about_centroid(state["T"], pts_cam, axis_vec, step)
            refresh()
            report()
            return False
        return _cb

    def cycle_step(vis_):
        state["step_idx"] = (state["step_idx"] + 1) % len(translate_steps)
        report()
        return False

    def reset(vis_):
        state["T"] = T_init.copy()
        refresh()
        report()
        return False

    def confirm(vis_):
        state["confirmed"] = True
        vis.close()
        return False

    def abort(vis_):
        state["confirmed"] = False
        vis.close()
        return False

    vis.register_key_callback(263, make_translate([-1, 0, 0]))  # left
    vis.register_key_callback(262, make_translate([1, 0, 0]))   # right
    vis.register_key_callback(264, make_translate([0, -1, 0]))  # down
    vis.register_key_callback(265, make_translate([0, 1, 0]))   # up
    vis.register_key_callback(267, make_translate([0, 0, -1]))  # page down
    vis.register_key_callback(266, make_translate([0, 0, 1]))   # page up

    vis.register_key_callback(ord("1"), make_rotate([-1, 0, 0]))
    vis.register_key_callback(ord("2"), make_rotate([1, 0, 0]))
    vis.register_key_callback(ord("3"), make_rotate([0, -1, 0]))
    vis.register_key_callback(ord("4"), make_rotate([0, 1, 0]))
    vis.register_key_callback(ord("5"), make_rotate([0, 0, -1]))
    vis.register_key_callback(ord("6"), make_rotate([0, 0, 1]))

    vis.register_key_callback(ord("F"), cycle_step)
    vis.register_key_callback(ord("R"), reset)
    vis.register_key_callback(257, confirm)  # enter
    vis.register_key_callback(256, abort)    # escape

    print(
        f"{window_title}: Left/Right/Down/Up arrows = translate X/Y, "
        "PageUp/PageDown = translate Z, 1-6 = rotate about X/Y/Z (about the "
        "cloud's own center), f = cycle step size, r = reset (back to what's "
        "shown now), Enter = confirm, Escape = abort/revert to the fallback pose."
    )
    report()

    vis.run()
    vis.destroy_window()

    return state["T"] if state["confirmed"] else T_fallback


def _median_nn_distance(source, target, T):
    """Median nearest-neighbor distance from source (transformed by T) to target."""
    pts = np.asarray(source.points)
    pts_world = (T[:3, :3] @ pts.T).T + T[:3, 3]
    kd = o3d.geometry.KDTreeFlann(target)
    dists = []
    for p in pts_world:
        k, idx, d2 = kd.search_knn_vector_3d(p, 1)
        if k > 0:
            dists.append(np.sqrt(d2[0]))
    return float(np.median(dists)) if dists else float("inf")


def refine_icp_multiscale(
    source,
    target,
    init,
    # (voxel_size, max_correspondence_distance) per stage, coarse -> fine.
    # Default is deliberately tight: manual_align_transform now always runs
    # first and does the coarse bootstrapping (a human can tell "this is the
    # right link" in a way a distance metric can't), so this function's job
    # is fine local polish only, not recovering from a far-off seed. A wide
    # capture radius here was confirmed (visually, by a human) to let ICP
    # lock onto a nearby-but-wrong attractor -- e.g. an adjacent link -- even
    # from an already-correct seed, while still reporting a low, deceptively
    # "good" residual. Its caller in main() still shows the result for human
    # confirm/reject (see manual_align_transform's T_fallback), since no
    # fixed radius can be proven safe against every mesh's geometry.
    stages=((0.005, 0.015), (0.0025, 0.006), (0.001, 0.002)),
    max_iteration=50,
):
    """Coarse-to-fine point-to-plane ICP: shrinking voxel size + correspondence distance."""
    current = init
    result = None
    for voxel, max_corr in stages:
        src_down = source.voxel_down_sample(voxel)
        result = o3d.pipelines.registration.registration_icp(
            src_down, target,
            max_correspondence_distance=max_corr,
            init=current,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
        )
        current = result.transformation
        print(
            f"  voxel={voxel:.3f} max_corr={max_corr:.3f}  "
            f"fitness={result.fitness:.3f}  inlier_rmse={result.inlier_rmse:.4f}"
        )
    return result


def main():
    calibration_dir = "calibration_files"

    with open("extrinsic_calibration.json", "r") as f:
        extrinsics = json.load(f)

    available = discover_calibration_serials(calibration_dir)
    serials = [s for s in available if s in extrinsics]
    missing = sorted(set(available) - set(extrinsics))
    if missing:
        print(f"Warning: skipping {missing} (no entry in extrinsic_calibration.json)")
    if not serials:
        raise RuntimeError(f"No calibration_files serials with extrinsics found in {calibration_dir}")

    print("Calibrating cameras:", serials)

    with np.load(os.path.join(calibration_dir, "robot_pcd.npz")) as data:
        robot_pcd = data['pcd']

    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(robot_pcd)
    mesh_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # red
    mesh_pcd.estimate_normals()
    mesh_pcd.orient_normals_consistent_tangent_plane(k=15)

    for serial in serials:
        print (f"Refining {serial}")

        serial_dir = os.path.join(calibration_dir, serial)

        T_wc = np.asarray(extrinsics[serial]['X_WC'], dtype=np.float64)

        with np.load(os.path.join(serial_dir, "depth.npz")) as data:
            depth = data['depth'] / 1000.0

        mask = np.array(Image.open(os.path.join(serial_dir, "mask.png")))[..., 3]

        pcd = depth2pcd(depth, serial, T_wc=None, mask=mask)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
        pcd = pcd.remove_radius_outlier(nb_points=25, radius=0.01)[0]

        T_manual = manual_align_transform(pcd, mesh_pcd, T_wc, window_title="Manual align")

        manual_err = _median_nn_distance(pcd, mesh_pcd, T_manual)
        print(f"Manual alignment residual: {manual_err * 100:.2f}cm median nearest-mesh distance")

        # Keep ICP's capture radius close to what manual alignment actually achieved --
        # a wide radius was confirmed to let ICP lock onto a nearby-but-wrong attractor
        # even from a good seed (see refine_icp_multiscale's docstring).
        stage1_radius = float(np.clip(manual_err * 2.0, 0.005, 0.03))
        stages = (
            (stage1_radius / 3.0, stage1_radius),
            (stage1_radius / 6.0, stage1_radius / 2.5),
            (stage1_radius / 15.0, stage1_radius / 6.0),
        )
        icp_result = refine_icp_multiscale(pcd, mesh_pcd, T_manual, stages=stages)

        icp_err = _median_nn_distance(pcd, mesh_pcd, icp_result.transformation)
        print(f"Auto-ICP residual: {icp_err * 100:.2f}cm  (manual alone was {manual_err * 100:.2f}cm)")
        print(
            "Nearest-mesh distance alone can't tell a correct match from a wrong-but-nearby "
            "one (e.g. locking onto an adjacent link) -- look at the window and confirm or "
            "adjust; Escape reverts to your manual-only alignment."
        )

        T_wc_refined = manual_align_transform(
            pcd, mesh_pcd, icp_result.transformation, T_fallback=T_manual,
            window_title="Confirm auto-ICP result",
        )

        extrinsics[serial]['X_WC'] = np.asarray(T_wc_refined).tolist()


    pc_list = []

    for serial in serials:

        serial_dir = os.path.join(calibration_dir, serial)

        with np.load(os.path.join(serial_dir, "depth.npz")) as data:
            depth = data['depth'] / 1000.0

        # Load the image
        color = np.array(Image.open(os.path.join(serial_dir, "color.png")))

        pcd= depth2pcd(depth, serial, color=color, T_wc=extrinsics[serial]['X_WC'])
        pc_list.append(pcd)

    merged_pc = o3d.geometry.PointCloud()
    for p in pc_list:
        merged_pc += p
    o3d.visualization.draw_geometries([merged_pc, mesh_pcd])

    with open("extrinsic_calibration.json", "w") as f:
        json.dump(extrinsics, f, indent=8)

    return serials


if __name__ == "__main__":
    serials = main()

    stream = MultiRealSenseStream(serials, "extrinsic_calibration.json")
    viewer = LivePointCloudViewer()

    for i in range (1000):
        pc_list = []
        datapoints = stream.get_datapoints()

        merged_pc, _ = get_fused_point_cloud(datapoints)

        #Convert to numpy
        pts = np.asarray(merged_pc.points)
        cols = np.asarray(merged_pc.colors) if merged_pc.has_colors() else None

        viewer.update(pts, cols)

    stream.stop()
