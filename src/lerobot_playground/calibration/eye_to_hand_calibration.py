import numpy as np
import cv2
import json

def write_camera_transform(serial, matrix, json_path):
    """
    serial: str       e.g. "234222302164"
    matrix: np.ndarray or list of lists, shape (4,4)
    json_path: str    where to save the JSON file
    """
    # Convert matrix to a standard Python list
    matrix_list = matrix.tolist() if hasattr(matrix, "tolist") else matrix

    data = {
        serial: {
            "X_WC": matrix_list
        }
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=8)

base_to_gripper_transform = np.load('calibration_data/base_to_gripper_transform.npy')
camera_to_target_transform = np.load('calibration_data/camera_to_target_transform.npy')

R_gripper2base = []
t_gripper2base = []
R_target2cam = []
t_target2cam = []

R_base2gripper = []
t_base2gripper = []
R_cam2target = []
t_cam2target = []

for T_bg, T_ct in zip(base_to_gripper_transform, camera_to_target_transform):
    # Invert transforms

    T_gb = np.linalg.inv(T_bg)  # gripper -> base
    T_tc = np.linalg.inv(T_ct)  # target -> camera

    # Extract R, t
    R_gripper2base.append(T_gb[:3, :3])
    t_gripper2base.append(T_gb[:3, 3])
    R_target2cam.append(T_tc[:3, :3])
    t_target2cam.append(T_tc[:3, 3])

    # Extract R, t
    R_base2gripper.append(T_bg[:3, :3])
    t_base2gripper.append(T_bg[:3, 3])
    R_cam2target.append(T_ct[:3, :3])
    t_cam2target.append(T_ct[:3, 3])

# T_gripper2base is the gripper in base coordinate system in OpenCV. In my code T_base2gripper is the gripper in base coordinate system. 
#OpenCV expects T_base2gripper, so we pass T_gripper2base

#OpenCV expects T_target2cam, which is target in camera coordinate system. 
R_base2cam, t_base2cam = cv2.calibrateHandEye(
    R_gripper2base=R_gripper2base,
    t_gripper2base=t_gripper2base,
    R_target2cam=R_target2cam,
    t_target2cam=t_target2cam,
)

T_base2cam = np.eye(4)
T_base2cam[:3, :3] = R_base2cam
T_base2cam[:3, 3] = t_base2cam.squeeze()

print(T_base2cam[:3, 3])

T_cam2base = np.linalg.inv(T_base2cam)

for T_base2gripper, T_cam2target in zip(base_to_gripper_transform,camera_to_target_transform):

    T_gripper2base = np.linalg.inv(T_base2gripper)
    T_target2cam = np.linalg.inv(T_cam2target)

    T_gripper2target = T_gripper2base @ T_base2cam @ T_target2cam

    print(T_gripper2target[:3,3])

breakpoint()
write_camera_transform("244622072067", T_base2cam, "extrinsic_calibration.json")
np.save('calibration_data/base_to_camera_transform.npy', T_base2cam)
