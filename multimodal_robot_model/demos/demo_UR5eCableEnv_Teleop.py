import numpy as np
import cv2
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
import pyspacemouse
import mujoco
from utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager

# Setup gym
env = gym.make(
  "multimodal_robot_model/UR5eCableEnv-v0",
  render_mode="human",
  extra_camera_configs=[{"name": "front", "size": (224, 224)}, {"name": "side", "size": (224, 224)}]
)
obs, info = env.reset(seed=42)

# Setup motion manager
motion_manager = MotionManager(env)

# Setup record manager
record_manager = RecordManager(env)

# Setup spacemouse
pyspacemouse.open()

reset = True
while True:
    # Reset
    if reset:
        motion_manager.reset()
        record_manager.reset()
        record_manager.setupSimWorld()
        obs, info = env.reset()
        print("== [UR5eCableEnv] data_idx: {}, world_idx: {} ==".format(record_manager.data_idx, record_manager.world_idx))
        print("- Press space key to start automatic grasping.")
        reset = False

    # Read spacemouse
    spacemouse_state = pyspacemouse.read()

    # Set arm command
    if record_manager.status == RecordStatus.PRE_REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 1.02 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
    elif record_manager.status == RecordStatus.REACH:
        target_pos = env.unwrapped.model.body("cable_end").pos.copy()
        target_pos[2] = 0.995 # [m]
        motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
    elif record_manager.status == RecordStatus.TELEOP:
        pos_scale = 1e-2
        delta_pos = pos_scale * np.array([-1.0 * spacemouse_state.y, spacemouse_state.x, spacemouse_state.z])
        rpy_scale = 5e-3
        delta_rpy = rpy_scale * np.array([-1.0 * spacemouse_state.roll, -1.0 * spacemouse_state.pitch, -2.0 * spacemouse_state.yaw])
        motion_manager.setRelativeTargetSE3(delta_pos, delta_rpy)

    # Set gripper command
    if record_manager.status == RecordStatus.GRASP:
        motion_manager.gripper_pos = env.action_space.high[6]
    elif record_manager.status == RecordStatus.TELEOP:
        gripper_scale = 5.0
        if spacemouse_state.buttons[0] > 0 and spacemouse_state.buttons[1] <= 0:
            motion_manager.gripper_pos += gripper_scale
        elif spacemouse_state.buttons[1] > 0 and spacemouse_state.buttons[0] <= 0:
            motion_manager.gripper_pos -= gripper_scale

    # Draw markers
    motion_manager.drawMarkers()

    # Solve IK
    motion_manager.inverseKinematics()

    # Step environment
    action = motion_manager.getAction()
    obs, _, _, _, info = env.step(action)

    # Record data
    if record_manager.status == RecordStatus.TELEOP:
        record_manager.appendSingleData(RecordKey.TIME, record_manager.status_elapsed_duration)
        record_manager.appendSingleData(RecordKey.JOINT, action)
        record_manager.appendSingleData(RecordKey.FRONT_IMAGE, info["images"]["front"])
        record_manager.appendSingleData(RecordKey.SIDE_IMAGE, info["images"]["side"])
        record_manager.appendSingleData(RecordKey.WRENCH, obs[16:])

    # Draw images
    window_image = np.concatenate([info["images"]["front"], info["images"]["side"], record_manager.getStatusImage()])
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
            print("- Press space key to start teleoperation after robot grasps the cable.")
    elif record_manager.status == RecordStatus.GRASP:
        if key == 32: # space key
            record_manager.goToNextStatus()
            print("- Press space key to finish teleoperation.")
    elif record_manager.status == RecordStatus.TELEOP:
        if key == 32: # space key
            print("- Press the 's' key if the teleoperation succeeded, or the 'f' key if it failed. (duration: {:.1f} [s])".format(
                record_manager.status_elapsed_duration))
            record_manager.goToNextStatus()
    elif record_manager.status == RecordStatus.END:
        if key == ord("s"):
            # Save data
            filename = "teleop_data/env{:0>1}/UR5eCableEnv_env{:0>1}_{:0>3}.npz".format(
                record_manager.world_idx, record_manager.world_idx, record_manager.data_idx)
            record_manager.saveData(filename)
            print("- Teleoperation succeeded: Save the data as {}".format(filename))
            reset = True
        elif key == ord("f"):
            print("- Teleoperation failed: Reset without saving")
            reset = True
    if key == 27: # escape key
        break

# env.close()
