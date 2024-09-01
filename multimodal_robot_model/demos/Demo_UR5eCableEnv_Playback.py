import argparse
import numpy as np
import cv2
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
from DemoUtils import MotionManager, RecordStatus, RecordKey, RecordManager, convertDepthImageToColorImage

parser = argparse.ArgumentParser()
parser.add_argument("teleop_filename")
parser.add_argument("--pole-pos-idx", type=int, default=None, help="index of the position of poles (0-5)")
args = parser.parse_args()

# Setup gym
env = gym.make(
  "multimodal_robot_model/UR5eCableEnv-v0",
  render_mode="human",
  extra_camera_configs=[
      {"name": "front", "size": (480, 640)},
      {"name": "side", "size": (480, 640)},
      {"name": "hand", "size": (480, 640)},
  ]
)
obs, info = env.reset(seed=42)

# Setup motion manager
motion_manager = MotionManager(env)

# Setup record manager
record_manager = RecordManager(env)
record_manager.loadData(args.teleop_filename)
pole_pos_idx = args.pole_pos_idx
if pole_pos_idx is None:
    pole_pos_idx = record_manager.data_seq["world_idx"].tolist()
record_manager.setupSimWorld(pole_pos_idx)

print("- Press the 'n' key to start automatic grasping.")

while True:
    # Set arm command
    if record_manager.status == RecordStatus.PRE_REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 1.02 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
    elif record_manager.status == RecordStatus.REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 0.995 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)

    # Set gripper command
    if record_manager.status == RecordStatus.GRASP:
        motion_manager.gripper_pos = env.action_space.high[6]
    elif record_manager.status == RecordStatus.TELEOP:
        motion_manager.gripper_pos = record_manager.getSingleData(RecordKey.ACTION, time_idx)[6]

    # Solve IK
    if record_manager.status == RecordStatus.PRE_REACH or record_manager.status == RecordStatus.REACH:
        motion_manager.inverseKinematics()
    elif record_manager.status == RecordStatus.TELEOP:
        motion_manager.joint_pos = record_manager.getSingleData(RecordKey.ACTION, time_idx)[:6]

    # Step environment
    action = motion_manager.getAction()
    _, _, _, _, info = env.step(action)

    # Draw images
    status_image = record_manager.getStatusImage()
    online_rgb_images = []
    online_depth_images = []
    for camera_name in ("front", "side", "hand"):
        rgb_image = info["rgb_images"][camera_name]
        image_ratio = rgb_image.shape[1] / rgb_image.shape[0]
        resized_image_size = (status_image.shape[1], int(status_image.shape[1] / image_ratio))
        online_rgb_images.append(cv2.resize(rgb_image, resized_image_size))
        depth_image = convertDepthImageToColorImage(info["depth_images"][camera_name])
        online_depth_images.append(cv2.resize(depth_image, resized_image_size))
    online_rgb_images.append(status_image)
    online_depth_images.append(np.full_like(status_image, 255))
    if record_manager.status == RecordStatus.TELEOP:
        record_rgb_images = []
        record_depth_images = []
        for record_rgb_key, record_depth_key in ((RecordKey.FRONT_RGB_IMAGE, RecordKey.FRONT_DEPTH_IMAGE),
                                                 (RecordKey.SIDE_RGB_IMAGE, RecordKey.SIDE_DEPTH_IMAGE),
                                                 (RecordKey.HAND_RGB_IMAGE, RecordKey.HAND_DEPTH_IMAGE)):
            rgb_image = record_manager.getSingleData(record_rgb_key, time_idx)
            image_ratio = rgb_image.shape[1] / rgb_image.shape[0]
            resized_image_size = (status_image.shape[1], int(status_image.shape[1] / image_ratio))
            record_rgb_images.append(cv2.resize(rgb_image, resized_image_size))
            depth_image = convertDepthImageToColorImage(record_manager.getSingleData(record_depth_key, time_idx))
            record_depth_images.append(cv2.resize(depth_image, resized_image_size))
        record_rgb_images.append(np.full_like(status_image, 255))
        record_depth_images.append(np.full_like(status_image, 255))

        window_image = cv2.hconcat((cv2.vconcat(online_rgb_images), cv2.vconcat(record_rgb_images),
                                    cv2.vconcat(online_depth_images), cv2.vconcat(record_depth_images)))
    else:
        window_image = cv2.hconcat((cv2.vconcat(online_rgb_images), cv2.vconcat(online_depth_images)))
    cv2.namedWindow("image", flags=(cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL))
    cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)

    # Manage status
    if record_manager.status == RecordStatus.INITIAL:
        if key == ord("n"):
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.PRE_REACH:
        pre_reach_duration = 0.7 # [s]
        if record_manager.status_elapsed_duration > pre_reach_duration:
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.REACH:
        reach_duration = 0.3 # [s]
        if record_manager.status_elapsed_duration > reach_duration:
            record_manager.goToNextStatus()
            print("- Press the 'n' key to start playback after robot grasps the cable.")
    elif record_manager.status == RecordStatus.GRASP:
        time_idx = 0
        if key == ord("n"):
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.TELEOP:
        time_idx += 1
        if time_idx == len(record_manager.data_seq["time"]):
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.END:
        if key == ord("n"):
            print("- Press the 'n' key to exit.")
            break
    if key == 27: # escape key
        break

# env.close()
