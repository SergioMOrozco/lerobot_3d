import open3d as o3d
from PIL import Image
import numpy as np
import json
import os
from lerobot_3d.point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from lerobot_3d.point_clouds.alignment_viewer import AlignmentViewer


def discover_calibration_serials(calib_dir):
    """Serials with both depth.npz and mask.png under calib_dir/<serial>/."""
    serials = []
    for name in sorted(os.listdir(calib_dir)):
        d = os.path.join(calib_dir, name)
        if (
            os.path.isdir(d)
            and os.path.exists(os.path.join(d, "depth.npz"))
            and os.path.exists(os.path.join(d, "mask.png"))
        ):
            serials.append(name)
    return serials


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
    # Default is deliberately tight: the manual-align step (AlignmentViewer.align) now
    # always runs first and does the coarse bootstrapping (a human can tell "this is the
    # right link" in a way a distance metric can't), so this function's job
    # is fine local polish only, not recovering from a far-off seed. A wide
    # capture radius here was confirmed (visually, by a human) to let ICP
    # lock onto a nearby-but-wrong attractor -- e.g. an adjacent link -- even
    # from an already-correct seed, while still reporting a low, deceptively
    # "good" residual. Its caller in main() still shows the result for human
    # confirm/reject (see AlignmentViewer.align's T_fallback), since no
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


def main(viewer: AlignmentViewer):
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

    viewer.show("/target", np.asarray(mesh_pcd.points), np.asarray(mesh_pcd.colors))

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

        T_manual = viewer.align(np.asarray(pcd.points), T_wc, title="Manual align")

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
            "one (e.g. locking onto an adjacent link) -- look at the viewer and confirm or "
            "adjust; Abort reverts to your manual-only alignment."
        )

        T_wc_refined = viewer.align(
            np.asarray(pcd.points), icp_result.transformation, T_fallback=T_manual,
            title="Confirm auto-ICP result",
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

    viewer.remove("/moving")
    viewer.show(
        "/merged",
        np.asarray(merged_pc.points),
        np.asarray(merged_pc.colors) if merged_pc.has_colors() else None,
    )
    viewer.wait_for_confirmation("Review merged result")

    with open("extrinsic_calibration.json", "w") as f:
        json.dump(extrinsics, f, indent=8)

    return serials


if __name__ == "__main__":
    viewer = AlignmentViewer()
    try:
        serials = main(viewer)

        stream = MultiRealSenseStream(serials, "extrinsic_calibration.json")
        try:
            for i in range(1000):
                datapoints = stream.get_datapoints()

                merged_pc, _ = get_fused_point_cloud(datapoints)

                pts = np.asarray(merged_pc.points)
                cols = np.asarray(merged_pc.colors) if merged_pc.has_colors() else None

                viewer.show("/live", pts, cols)
        finally:
            stream.stop()
    finally:
        viewer.close()
