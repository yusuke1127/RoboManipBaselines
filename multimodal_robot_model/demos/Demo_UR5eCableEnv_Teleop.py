import argparse
import numpy as np
import cv2
import gymnasium as gym
import multimodal_robot_model
import pinocchio as pin
import pyspacemouse
from Utils_UR5eCableEnv import MotionManager, RecordStatus, RecordKey, RecordManager, \
    convertDepthImageToColorImage, convertDepthImageToPointCloud

parser = argparse.ArgumentParser()
parser.add_argument("--enable-3d-plot", action="store_true", help="whether to enable 3d plot")
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
record_manager.setupCameraInfo((RecordKey.FRONT_DEPTH_IMAGE, RecordKey.SIDE_DEPTH_IMAGE, RecordKey.HAND_DEPTH_IMAGE))

# Setup spacemouse
pyspacemouse.open()

# Setup 3D plot
if args.enable_3d_plot:
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams["keymap.quit"] = ["q", "escape"]
    fig, ax = plt.subplots(len(env.unwrapped.cameras), 1, subplot_kw=dict(projection="3d"))
    fig.tight_layout()
    scatter_list = [None] * len(env.unwrapped.cameras)

reset = True
while True:
    # Reset
    if reset:
        motion_manager.reset()
        record_manager.reset()
        record_manager.setupSimWorld()
        obs, info = env.reset()
        print("== [UR5eCableEnv] data_idx: {}, world_idx: {} ==".format(record_manager.data_idx, record_manager.world_idx))
        print("- Press the 'n' key to start automatic grasping.")
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

    # Get action
    action = motion_manager.getAction()

    # Record data
    if record_manager.status == RecordStatus.TELEOP:
        record_manager.appendSingleData(RecordKey.TIME, record_manager.status_elapsed_duration)
        record_manager.appendSingleData(RecordKey.JOINT_POS, motion_manager.getJointPos(obs))
        record_manager.appendSingleData(RecordKey.JOINT_VEL, motion_manager.getJointVel(obs))
        record_manager.appendSingleData(RecordKey.FRONT_RGB_IMAGE, info["rgb_images"]["front"])
        record_manager.appendSingleData(RecordKey.SIDE_RGB_IMAGE, info["rgb_images"]["side"])
        record_manager.appendSingleData(RecordKey.HAND_RGB_IMAGE, info["rgb_images"]["hand"])
        record_manager.appendSingleData(RecordKey.FRONT_DEPTH_IMAGE, info["depth_images"]["front"])
        record_manager.appendSingleData(RecordKey.SIDE_DEPTH_IMAGE, info["depth_images"]["side"])
        record_manager.appendSingleData(RecordKey.HAND_DEPTH_IMAGE, info["depth_images"]["hand"])
        record_manager.appendSingleData(RecordKey.WRENCH, obs[16:])
        record_manager.appendSingleData(RecordKey.ACTION, action)

    # Step environment
    obs, _, _, _, info = env.step(action)

    # Draw images
    status_image = record_manager.getStatusImage()
    rgb_images = []
    depth_images = []
    for camera_name in ("front", "side", "hand"):
        rgb_image = info["rgb_images"][camera_name]
        image_ratio = rgb_image.shape[1] / rgb_image.shape[0]
        resized_image_size = (status_image.shape[1], int(status_image.shape[1] / image_ratio))
        rgb_images.append(cv2.resize(rgb_image, resized_image_size))
        depth_image = convertDepthImageToColorImage(info["depth_images"][camera_name])
        depth_images.append(cv2.resize(depth_image, resized_image_size))
    rgb_images.append(status_image)
    depth_images.append(np.full_like(status_image, 255))
    window_image = cv2.hconcat((cv2.vconcat(rgb_images), cv2.vconcat(depth_images)))
    cv2.namedWindow("image", flags=(cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL))
    cv2.imshow("image", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)

    # Draw point clouds
    if args.enable_3d_plot:
        dist_thre_list = (3.0, 3.0, 0.8) # [m]
        for camera_idx, camera_name in enumerate(("front", "side", "hand")):
            point_cloud_skip = 10
            small_depth_image = info["depth_images"][camera_name][::point_cloud_skip, ::point_cloud_skip]
            small_rgb_image = info["rgb_images"][camera_name][::point_cloud_skip, ::point_cloud_skip]
            fovy = env.unwrapped.model.cam_fovy[camera_idx]
            xyz_array, rgb_array = convertDepthImageToPointCloud(
                small_depth_image, fovy=fovy, rgb_image=small_rgb_image, dist_thre=dist_thre_list[camera_idx])
            if scatter_list[camera_idx] is None:
                get_min_max = lambda v_min, v_max: (0.75 * v_min + 0.25 * v_max, 0.25 * v_min + 0.75 * v_max)
                ax[camera_idx].view_init(elev=-90, azim=-90)
                ax[camera_idx].set_xlim(*get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max()))
                ax[camera_idx].set_ylim(*get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max()))
                ax[camera_idx].set_zlim(*get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max()))
            else:
                scatter_list[camera_idx].remove()
            ax[camera_idx].axis("off")
            ax[camera_idx].set_aspect("equal")
            scatter_list[camera_idx] = ax[camera_idx].scatter(
                xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array)
        plt.draw()
        plt.pause(0.001)

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
            print("- Press the 'n' key to start teleoperation after robot grasps the cable.")
    elif record_manager.status == RecordStatus.GRASP:
        if key == ord("n"):
            record_manager.goToNextStatus()
            print("- Press the 'n' key to finish teleoperation.")
    elif record_manager.status == RecordStatus.TELEOP:
        if key == ord("n"):
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
