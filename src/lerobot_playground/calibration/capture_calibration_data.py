from pathlib import Path

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from urchin import URDF
import numpy as np
import cv2

cap = cv2.VideoCapture(4)

def map_joint_value(raw, raw_range, phys_range):
    return phys_range[0] + (raw - raw_range[0]) / (raw_range[1] - raw_range[0]) * (phys_range[1] - phys_range[0])

raw_ranges = {
    'shoulder_pan': [-100, 100],
    'shoulder_lift': [-100, 100],
    'elbow_flex': [-100, 100],
    'wrist_flex': [-100, 100],
    'wrist_roll': [-100, 100],
    'gripper': [0, 100] 
}
phys_ranges = {
    'shoulder_pan': [-1.91986, 1.91986],
    'shoulder_lift': [-1.74533, 1.74533],
    'elbow_flex': [-1.69, 1.69],
    'wrist_flex': [-1.65806, 1.65806],
    'wrist_roll': [-2.74385, 2.84121],
    'gripper': [-0.174533, 1.74533]
}

# Load your URDF file (this script lives next to ``so101_new_calib.urdf`` in the package).
_CALIB_DIR = Path(__file__).resolve().parent
robot_urdf = URDF.load(str(_CALIB_DIR / "so101_new_calib.urdf"))

# Define your joint angles in radians
# Example order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

# Compute the forward kinematics to the end effector
link_name = 'gripper_frame_link'


robot_1_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="bender_follower_arm",
)

teleop_1_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="bender_leader_arm",
)

robot_2_config = SO101FollowerConfig(
    port="/dev/ttyACM3",
    id="clamps_follower_arm",
)

teleop_2_config = SO101LeaderConfig(
    port="/dev/ttyACM2",
    id="clamps_leader_arm",
)

robot_1 = SO101Follower(robot_1_config)
robot_2 = SO101Follower(robot_2_config)
teleop_1_device = SO101Leader(teleop_1_config)
teleop_2_device = SO101Leader(teleop_2_config)
robot_1.connect()
robot_2.connect()
teleop_1_device.connect()
teleop_2_device.connect()



num = 0

base_to_gripper_transforms = []

while True:
    action = teleop_1_device.get_action()
    robot_1.send_action(action)

    action = teleop_2_device.get_action()
    robot_2.send_action(action)

    j = robot_1.get_observation()

    joint_positions = {
        'shoulder_pan': map_joint_value(j['shoulder_pan.pos'], raw_ranges['shoulder_pan'], phys_ranges['shoulder_pan']),
        'shoulder_lift': map_joint_value(j['shoulder_lift.pos'], raw_ranges['shoulder_lift'], phys_ranges['shoulder_lift']),
        'elbow_flex': map_joint_value(j['elbow_flex.pos'], raw_ranges['elbow_flex'], phys_ranges['elbow_flex']),
        'wrist_flex': map_joint_value(j['wrist_flex.pos'], raw_ranges['wrist_flex'], phys_ranges['wrist_flex']),
        'wrist_roll': map_joint_value(j['wrist_roll.pos'], raw_ranges['wrist_roll'], phys_ranges['wrist_roll']),
        'gripper': map_joint_value(j['gripper.pos'], raw_ranges['gripper'], phys_ranges['gripper'])
    }

    T = robot_urdf.link_fk(cfg=joint_positions)[robot_urdf.link_map[link_name]]

    print("Position:", T[:3, 3])

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('calibration_data/images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

        base_to_gripper_transforms.append(T)

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()

np.save('calibration_data/base_to_gripper_transform.npy', np.array(base_to_gripper_transforms))
