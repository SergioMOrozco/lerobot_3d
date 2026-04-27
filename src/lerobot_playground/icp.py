import open3d as o3d
from PIL import Image
import numpy as np
import json
import os
from lerobot_playground.point_clouds.camera_stream import MultiRealSenseStream, get_fused_point_cloud
from lerobot_playground.point_clouds.point_cloud_viewer import LivePointCloudViewer

def depth2pcd(depth, serial, color = None, T_wc= None, mask = None):

    with open("intrinsic_calibration.json", "r") as f:
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

    if color is not None:

        if color.dtype == np.float32:
            img_uint8 = np.array(color * 255, dtype=np.uint8)
        else:
            img_uint8 = np.array(color)

        color_image = o3d.geometry.Image(img_uint8)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False
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

serials = ["044322073544", "244622072067"]

with open("extrinsic_calibration.json", "r") as f:
    extrinsics = json.load(f)

calibration_dir = "calibration_files"

with np.load(os.path.join(calibration_dir, "robot_pcd.npz")) as data:
    robot_pcd = data['pcd']

mesh_pcd = o3d.geometry.PointCloud()
mesh_pcd.points = o3d.utility.Vector3dVector(robot_pcd)
mesh_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # red
mesh_pcd.estimate_normals()

for serial in serials:
    print (f"Refining {serial}")

    serial_dir = os.path.join(calibration_dir, serial)

    T_wc = extrinsics[serial]['X_WC']

    with np.load(os.path.join(serial_dir, "depth.npz")) as data:
        depth = data['depth'] / 1000.0

    # Load the image
    mask = np.array(Image.open(os.path.join(serial_dir, "mask.png")))[..., 3]
    img = np.array(Image.open(os.path.join(serial_dir, "color.png")))

    pcd = depth2pcd(depth, serial, T_wc=None, mask=mask)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
    pcd = pcd.remove_radius_outlier(nb_points=25, radius=0.01)[0]

    threshold = 0.01  # 2 cm, tune
    icp_result = o3d.pipelines.registration.registration_icp(
        source=pcd,             # observed point cloud
        target=mesh_pcd,        # model point cloud (robot mesh)
        max_correspondence_distance=threshold,
        init=T_wc,              # initial guess
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    pcd = depth2pcd(depth, serial, T_wc=T_wc, mask=mask)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
    pcd = pcd.remove_radius_outlier(nb_points=25, radius=0.01)[0]

    o3d.visualization.draw_geometries([mesh_pcd, pcd])

    T_wc_refined = icp_result.transformation

    extrinsics[serial]['X_WC'] = T_wc_refined

    pcd = depth2pcd(depth, serial, T_wc=T_wc_refined, mask=mask)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
    pcd = pcd.remove_radius_outlier(nb_points=25, radius=0.01)[0]

    o3d.visualization.draw_geometries([mesh_pcd, pcd])


pc_list = []

for serial in serials:

    serial_dir = os.path.join(calibration_dir, serial)

    with np.load(os.path.join(serial_dir, "depth.npz")) as data:
        #depth = data['depth'] / 1000.0
        depth = data['depth']

    # Load the image
    color = np.array(Image.open(os.path.join(serial_dir, "color.png")))

    pcd= depth2pcd(depth, serial, color=color, T_wc=extrinsics[serial]['X_WC'])
    pc_list.append(pcd)

merged_pc = o3d.geometry.PointCloud()
for p in pc_list:
    merged_pc += p
o3d.visualization.draw_geometries([merged_pc, mesh_pcd])

new_extrinsics = {}

for serial in serials:

    matrix_list = extrinsics[serial]['X_WC'].tolist()

    new_extrinsics[serial] = {
        "X_WC": matrix_list
    }

with open("extrinsic_calibration.json", "w") as f:
    json.dump(new_extrinsics, f, indent=8)

if __name__ == "__main__":
   serials = ["244622072067", "044322073544"]
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