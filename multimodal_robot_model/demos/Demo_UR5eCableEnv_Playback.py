import sys
import argparse
import numpy as np
import cv2
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
from Utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager

parser = argparse.ArgumentParser()
parser.add_argument("teleop_filename")
parser.add_argument("--pole-pos-idx", type=int, default=None, help="index of the position of poles (0-5)")
args = parser.parse_args()

# Setup gym
env = gym.make(
  "multimodal_robot_model/UR5eCableEnv-v0",
  render_mode="human",
  extra_camera_configs=[{"name": "front", "size": (480, 640)}, {"name": "side", "size": (480, 640)}]
)
obs, info = env.reset(seed=42)

# Setup motion manager
motion_manager = MotionManager(env)

# Setup record manager
record_manager = RecordManager(env)
record_manager.loadData(args.teleop_filename)
pole_pos_idx = args.pole_pos_idx
if pole_pos_idx is None:
    pole_pos_idx = record_manager.data_seq["pole_pos_idx"].tolist()
record_manager.setupSimWorld(pole_pos_idx)

print("- Press space key to start automatic grasping.")

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
        motion_manager.gripper_pos = record_manager.getSingleData(RecordKey.JOINT, time_idx)[6]

    # Solve IK
    if record_manager.status == RecordStatus.PRE_REACH or record_manager.status == RecordStatus.REACH:
        motion_manager.inverseKinematics()
    elif record_manager.status == RecordStatus.TELEOP:
        motion_manager.joint_pos = record_manager.getSingleData(RecordKey.JOINT, time_idx)[:6]

    # Step environment
    action = motion_manager.getAction()
    _, _, _, _, info = env.step(action)

    # Draw images
    status_image = record_manager.getStatusImage()
    front_image_ratio = float(info["images"]["front"].shape[1]) / info["images"]["front"].shape[0]
    side_image_ratio = float(info["images"]["side"].shape[1]) / info["images"]["side"].shape[0]
    online_image = cv2.vconcat([cv2.resize(info["images"]["front"], (224, int(224 / front_image_ratio))),
                                cv2.resize(info["images"]["side"], (224, int(224 / side_image_ratio))),
                                status_image])
    # TODO: Fix slow access to record images
    # if record_manager.status == RecordStatus.TELEOP:
    #     record_image = cv2.vconcat([record_manager.getSingleData(RecordKey.FRONT_IMAGE, time_idx),
    #                                 record_manager.getSingleData(RecordKey.SIDE_IMAGE, time_idx),
    #                                 np.full_like(status_image, 255)])
    # else:
    #     record_image = np.full_like(online_image, 255)
    # window_image = cv2.hconcat([online_image, record_image])
    window_image = online_image
    cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)

    # Manage status
    if record_manager.status == RecordStatus.INITIAL:
        if key == 32: # space key
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.PRE_REACH:
        pre_reach_duration = 0.7 # [s]
        if record_manager.status_elapsed_duration > pre_reach_duration:
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.REACH:
        reach_duration = 0.3 # [s]
        if record_manager.status_elapsed_duration > reach_duration:
            record_manager.goToNextStatus()
            print("- Press space key to start playback after robot grasps the cable.")
    elif record_manager.status == RecordStatus.GRASP:
        time_idx = 0
        if key == 32: # space key
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.TELEOP:
        time_idx += 1
        if time_idx == len(record_manager.data_seq["time"]):
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.END:
        if key == 32: # space key
            print("- Press space key to exit.")
            break
    if key == 27: # escape key
        break

# env.close()
