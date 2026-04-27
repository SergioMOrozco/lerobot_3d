#!/usr/bin/env python3
import os
import json
import numpy as np
from PIL import Image
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------
# Depth -> point cloud in camera frame
# ------------------------------------------------------------
def depth2pcd(depth, serial, mask=None):
    with open("intrinsic_calibration.json", "r") as f:
        intr = json.load(f)[serial]

    fx, fy = intr["fl_x"], intr["fl_y"]
    cx, cy = intr["cx"], intr["cy"]
    w, h = intr["w"], intr["h"]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    if mask is not None:
        depth = depth.copy()
        depth[mask == 0] = 0.0

    depth_img = o3d.geometry.Image(depth.astype(np.float32))

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_img,
        intrinsic,
        depth_scale=1.0,
    )

    return pcd


# ------------------------------------------------------------
# Fixed camera→mesh correspondences (indices into cam_pts and mesh_pts)
# ------------------------------------------------------------
def compute_fixed_cam_mesh_correspondences(pts_cam, T_init, mesh_kd, mesh_pts, max_dist=0.03):
    """
    pts_cam: (N,3) in camera frame
    T_init: 4x4 world_from_cam initial transform
    Returns:
        cam_idx: indices into pts_cam
        mesh_idx: indices into mesh_pts
    """
    R0, t0 = T_init[:3, :3], T_init[:3, 3]
    pts_world = (R0 @ pts_cam.T).T + t0

    cam_idx = []
    mesh_idx = []
    max_d2 = max_dist * max_dist

    for i, p in enumerate(pts_world):
        k, idx, d2 = mesh_kd.search_knn_vector_3d(p, 1)
        if k == 0:
            continue
        if d2[0] > max_d2:
            continue
        cam_idx.append(i)
        mesh_idx.append(idx[0])

    return np.asarray(cam_idx, dtype=np.int64), np.asarray(mesh_idx, dtype=np.int64)


# ------------------------------------------------------------
# Fixed camera↔camera correspondences (indices into cam_pts)
# ------------------------------------------------------------
def compute_fixed_cam_cam_pairs(pts1_cam, pts2_cam, T1_init, T2_init,
                                max_dist=0.05, max_pairs=2000):
    """
    pts1_cam: (N1,3) camera 1 frame
    pts2_cam: (N2,3) camera 2 frame
    Returns:
        idx1_pairs, idx2_pairs (same length)
        such that pts1_cam[idx1] ↔ pts2_cam[idx2]
    """
    R1, t1 = T1_init[:3, :3], T1_init[:3, 3]
    R2, t2 = T2_init[:3, :3], T2_init[:3, 3]

    pts1_world = (R1 @ pts1_cam.T).T + t1
    pts2_world = (R2 @ pts2_cam.T).T + t2

    pc2_world = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts2_world))
    kd2 = o3d.geometry.KDTreeFlann(pc2_world)

    idx1_pairs = []
    idx2_pairs = []

    max_d2 = max_dist * max_dist

    n1 = pts1_world.shape[0]
    if n1 == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    N = min(max_pairs, n1)
    sample_idx = np.random.choice(n1, size=N, replace=False)

    for i in sample_idx:
        p = pts1_world[i]
        k, idx, d2 = kd2.search_knn_vector_3d(p, 1)
        if k == 0:
            continue
        if d2[0] > max_d2:
            continue

        j = idx[0]
        idx1_pairs.append(i)
        idx2_pairs.append(j)

    return np.asarray(idx1_pairs, dtype=np.int64), np.asarray(idx2_pairs, dtype=np.int64)


# ------------------------------------------------------------
# Param <-> transform helpers
# ------------------------------------------------------------
def params_to_transforms(p):
    """
    p: (12,) = [r1(3), t1(3), r2(3), t2(3)]
    """
    r1 = p[0:3]
    t1 = p[3:6]
    r2 = p[6:9]
    t2 = p[9:12]

    T1 = np.eye(4)
    T1[:3, :3] = R.from_rotvec(r1).as_matrix()
    T1[:3, 3] = t1

    T2 = np.eye(4)
    T2[:3, :3] = R.from_rotvec(r2).as_matrix()
    T2[:3, 3] = t2

    return T1, T2


def matrix_to_rotvec(T):
    return R.from_matrix(T[:3, :3]).as_rotvec()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    calib_dir = "calibration_files"

    # -------- Load extrinsics --------
    with open("extrinsic_calibration.json", "r") as f:
        extr = json.load(f)

    serials = sorted(extr.keys())
    if len(serials) != 2:
        raise RuntimeError(f"Expected 2 cameras, found {len(serials)}: {serials}")

    s1, s2 = serials
    print("Calibrating cameras:", s1, s2)

    # -------- Load robot mesh --------
    mesh_npz = os.path.join(calib_dir, "robot_pcd.npz")
    with np.load(mesh_npz) as data:
        mesh_pts = data["pcd"]

    mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_pts))
    mesh = mesh.voxel_down_sample(0.005)
    mesh.estimate_normals()

    mesh_points = np.asarray(mesh.points)
    mesh_normals = np.asarray(mesh.normals)
    mesh_kd = o3d.geometry.KDTreeFlann(mesh)

    # -------- Load camera point clouds (ONCE, fixed) --------
    cam_pts = {}

    for serial in serials:
        dfile = os.path.join(calib_dir, serial, "depth.npz")
        mfile = os.path.join(calib_dir, serial, "mask.png")

        with np.load(dfile) as data:
            depth = data["depth"].astype(np.float32) / 1000.0  # mm->m

        mask = np.array(Image.open(mfile))[..., 3]

        pcd = depth2pcd(depth, serial, mask=mask)
        pcd = pcd.voxel_down_sample(0.004)

        pts = np.asarray(pcd.points)
        cam_pts[serial] = pts.copy()

        print(f"{serial}: using {pts.shape[0]} fixed points")

    # -------- Initial parameters --------
    T1_init = np.asarray(extr[s1]["X_WC"])
    T2_init = np.asarray(extr[s2]["X_WC"])

    params0 = np.zeros(12, dtype=np.float64)
    params0[0:3] = matrix_to_rotvec(T1_init)
    params0[3:6] = T1_init[:3, 3]
    params0[6:9] = matrix_to_rotvec(T2_init)
    params0[9:12] = T2_init[:3, 3]

    # -------- Fixed cam->mesh correspondences --------
    cam_idx1, mesh_idx1 = compute_fixed_cam_mesh_correspondences(
        cam_pts[s1], T1_init, mesh_kd, mesh_points
    )
    cam_idx2, mesh_idx2 = compute_fixed_cam_mesh_correspondences(
        cam_pts[s2], T2_init, mesh_kd, mesh_points
    )

    print(f"{s1}: {len(cam_idx1)} cam->mesh correspondences")
    print(f"{s2}: {len(cam_idx2)} cam->mesh correspondences")

    # -------- Fixed cam<->cam correspondences (index pairs into cam_pts) --------
    idx12_1, idx12_2 = compute_fixed_cam_cam_pairs(
        cam_pts[s1], cam_pts[s2], T1_init, T2_init
    )
    print(f"Camera-camera pairs: {idx12_1.shape[0]}")

    # Weights
    w_icp = 1.0
    w_cc = 0.1  # camera-camera consistency weight

    # -------- Residual function --------
    def residuals(p):
        T1, T2 = params_to_transforms(p)
        R1, t1 = T1[:3, :3], T1[:3, 3]
        R2, t2 = T2[:3, :3], T2[:3, 3]

        # Transform camera points to world
        p1w = (R1 @ cam_pts[s1].T).T + t1  # (N1,3)
        p2w = (R2 @ cam_pts[s2].T).T + t2  # (N2,3)

        # --- camera->mesh ICP residuals (point-to-plane) ---
        q1 = mesh_points[mesh_idx1]
        n1 = mesh_normals[mesh_idx1]
        r1 = np.sum(n1 * (p1w[cam_idx1] - q1), axis=1)

        q2 = mesh_points[mesh_idx2]
        n2 = mesh_normals[mesh_idx2]
        r2 = np.sum(n2 * (p2w[cam_idx2] - q2), axis=1)

        # --- camera<->camera consistency (3D diffs at index pairs) ---
        if idx12_1.size > 0:
            p1_pair = p1w[idx12_1]
            p2_pair = p2w[idx12_2]
            r12 = (p1_pair - p2_pair).reshape(-1)
        else:
            r12 = np.zeros(0, dtype=np.float64)

        return np.concatenate([w_icp * r1, w_icp * r2, w_cc * r12])

    # -------- Optimize --------
    print("Running joint ICP optimization...")
    result = least_squares(
        residuals,
        params0,
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=200,
        verbose=2,
    )
    print("Optimization finished.")
    print("Initial residual norm:", np.linalg.norm(residuals(params0)))
    print("Final   residual norm:", np.linalg.norm(residuals(result.x)))

    p_opt = result.x
    T1_opt, T2_opt = params_to_transforms(p_opt)

    # -------- Save updated extrinsics --------
    new_extr = {
        s1: {"X_WC": T1_opt.tolist()},
        s2: {"X_WC": T2_opt.tolist()},
    }

    with open("extrinsic_calibration.json", "w") as f:
        json.dump(new_extr, f, indent=4)

    print("Saved refined extrinsics to extrinsic_calibration.json")

    # -------- Visualize merged point clouds (using same cam_pts) --------
    merged = o3d.geometry.PointCloud()

    for serial, T in zip([s1, s2], [T1_opt, T2_opt]):
        Rw, tw = T[:3, :3], T[:3, 3]
        pts_cam = cam_pts[serial]
        pts_world = (Rw @ pts_cam.T).T + tw
        pcd_world = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
        merged += pcd_world

    o3d.visualization.draw_geometries([merged, mesh])


if __name__ == "__main__":
    main()
