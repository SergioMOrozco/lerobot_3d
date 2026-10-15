● On the client (Ubuntu 22.04), do these steps once, with the Ethernet cable plugged in between the client and the Jetson.

  1. Copy the setup files from the Jetson. This goes over Wi-Fi SSH, which still works:
  scp yam@128.148.140.225:yamming/config/cyclonedds.client.xml ~/
  scp yam@128.148.140.225:yamming/scripts/08_install_ros2.sh ~/

  2. Install ROS 2 Humble and CycloneDDS (skip if ROS 2 Humble is already installed):
  bash ~/08_install_ros2.sh
  sudo apt install -y ros-humble-rmw-cyclonedds-cpp ros-humble-image-transport-plugins
  # optional, for RViz and image viewers:
  sudo apt install -y ros-humble-desktop

  3. Give the cable port a static IP. Find the port name first with ip -br link. It's usually eno1, enp… or eth0, not the wl… one, which is Wi-Fi.
  IFACE=<your-ethernet-port>
  sudo nmcli con add type ethernet con-name yam-ethernet ifname $IFACE \
    ipv4.method manual ipv4.addresses 192.168.50.2/24 ipv4.never-default yes
  sudo nmcli con up yam-ethernet
  ping -c 3 192.168.50.1        # must reply (the Jetson must have run setup_ethernet.sh)

  4. Enlarge the network buffers, needed for camera images:
  printf 'net.core.rmem_max=2147483647\nnet.core.wmem_max=2147483647\nnet.ipv4.ipfrag_time=3\nnet.ipv4.ipfrag_high_thresh=134217728\n' \
    | sudo tee /etc/sysctl.d/60-yam-dds.conf && sudo sysctl --system

  5. Set up the shell. Add these to ~/.bashrc so every terminal has them:
  source /opt/ros/humble/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file://$HOME/cyclonedds.client.xml
  export ROS_DOMAIN_ID=42

  Check it works while the launch is running on the Jetson. Open a new terminal first so the settings load.
  ros2 topic list                                   # /left/joint_states, /right/cmd_joint, /camera/...
  ros2 topic hz /left/joint_states                  # about 50 Hz
  ros2 topic echo --once /right/joint_states
  ros2 topic hz /camera/left_wrist/color/image_raw/compressed

  Send a small test command with someone at the rig. Read the joint's current position first, then send a target close to it, because commands more than 0.35 rad from the current position are rejected:
  ros2 topic echo --once /left/joint_states        # note left_joint1's position, e.g. 0.01
  ros2 topic pub --once /left/cmd_joint sensor_msgs/msg/JointState \
    "{name: [left_joint1], position: [0.06]}"       # about 0.05 rad from where it is
  ros2 topic pub --once /left/cmd_joint sensor_msgs/msg/JointState \
    "{name: [left_gripper], position: [0.5]}"       # half open

  To look at images, run ros2 run rqt_image_view rqt_image_view and pick a /compressed topic.

  If ros2 topic list shows nothing:
  - Check that ping 192.168.50.1 works.
  - Check the Jetson ran setup_ethernet.sh.
  - Make sure the two settings that must match are the same on both machines: ROS_DOMAIN_ID=42 and RMW_IMPLEMENTATION=rmw_cyclonedds_cpp.
  - Run ros2 daemon stop and try again. The topic list is cached by a background process.

  None of this has been tested over the cable yet, so if anything fails, paste the output.