import os
import json
import cv2
import numpy as np

def save_intrinsics_json(serial, K, json_path):
    """
    serial: str (camera serial number)
    K: (3,3) numpy array (intrinsics matrix)
    json_path: path to output JSON file
    """
    # Load existing JSON (if any)
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Convert K to list so JSON can store it
    K_list = K.tolist()

    # Store under the serial number
    data[serial] = {"K": K_list}

    # Write back to disk
    with open(json_path, "w") as f:
        json.dump(data, f, indent=8)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 1. Define AprilTag pattern parameters (e.g., tag size, spacing if using a board)
# (column,row)
pattern_size = (3,4)
square_size_mm = 15  # Example tag size in mm
square_size_m = square_size_mm / 1000
# ... define world points based on your specific AprilTag layout

# 2. Capture or load calibration images
image_paths = [
    "calibration_data/images/244622072067/0.png",
    "calibration_data/images/244622072067/1.png",
    "calibration_data/images/244622072067/2.png",
    "calibration_data/images/244622072067/3.png",
    "calibration_data/images/244622072067/4.png",
    "calibration_data/images/244622072067/5.png",
    "calibration_data/images/244622072067/6.png",
    "calibration_data/images/244622072067/7.png",
    "calibration_data/images/244622072067/8.png",
    "calibration_data/images/244622072067/9.png",
    "calibration_data/images/244622072067/10.png",
    "calibration_data/images/244622072067/11.png",
    "calibration_data/images/244622072067/12.png",
    "calibration_data/images/244622072067/13.png",
    "calibration_data/images/244622072067/14.png",
    "calibration_data/images/244622072067/15.png",
    "calibration_data/images/244622072067/16.png",
    "calibration_data/images/244622072067/17.png",
    "calibration_data/images/244622072067/18.png",
    "calibration_data/images/244622072067/19.png",
    "calibration_data/images/244622072067/20.png",
    "calibration_data/images/244622072067/21.png",

] # List of paths to your calibration images

# Store detected image points and corresponding world points
obj_points = []  # 3D points in world coordinate system
img_points = []  # 2D points in image plane

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

cols, rows = pattern_size  # for readability
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        obj_points.append(objp)

        #corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        corners2 = corners
        img_points.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 5. Perform camera calibration
# Assuming you have a rough initial guess for image size for the calibration function
img_size = gray.shape[::-1] # (width, height)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img_size, None, None
)

camera_to_target_transforms = []
for rvec,tvec in zip(rvecs,tvecs):
    R, _ = cv2.Rodrigues(rvec)

    # Build the 4x4 transformation matrix
    # and invert to get camera_to_target
    T_camera_to_target = np.eye(4)
    T_camera_to_target[:3, :3] = R.T
    T_camera_to_target[:3, 3] = -R.T @ tvec.flatten()

    camera_to_target_transforms.append(T_camera_to_target)

#print("Camera Matrix (Intrinsics):\n", mtx)
#print("Distortion Coefficients:\n", dist)

for tvec in tvecs:
    print("Position: \n", tvec)

mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(obj_points)) )

np.save('calibration_data/camera_to_target_transform.npy', np.array(camera_to_target_transforms))
np.save('calibration_data/camera_instrinsic.npy', np.array(mtx))
np.save('calibration_data/camera_distortion_coefficients.npy', np.array(dist))


breakpoint()
save_intrinsics_json("244622072067", mtx, "intrinsic_calibration.json")