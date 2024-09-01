import argparse
import time
import numpy as np
import cv2
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import pyspacemouse
from DemoUtils import MotionManager, RecordStatus, RecordKey, RecordManager, \
    convertDepthImageToColorImage, convertDepthImageToPointCloud

class DemoTeleopBase(object):
    def __init__(self, env, demo_name):
        parser = argparse.ArgumentParser()
        parser.add_argument("--enable-3d-plot", action="store_true", help="whether to enable 3d plot")
        parser.add_argument("--compress-rgb", type=int, default=1, help="whether to compress rgb image")
        parser.add_argument("--compress-depth", type=int, default=0, help="whether to compress depth image (slow)")
        parser.add_argument("--replay-log", type=str, default=None, help="log file path when replay log motion")
        self.args = parser.parse_args()

        # Setup gym
        self.env = env
        self.env.reset(seed=42)
        self.demo_name = demo_name

        # Setup motion manager
        self.motion_manager = MotionManager(self.env)

        # Setup record manager
        self.camera_names = ("front", "side", "hand")
        self.rgb_keys = (RecordKey.FRONT_RGB_IMAGE, RecordKey.SIDE_RGB_IMAGE, RecordKey.HAND_RGB_IMAGE)
        self.depth_keys = (RecordKey.FRONT_DEPTH_IMAGE, RecordKey.SIDE_DEPTH_IMAGE, RecordKey.HAND_DEPTH_IMAGE)
        self.record_manager = RecordManager(self.env)
        self.record_manager.setupCameraInfo(self.camera_names, self.depth_keys)

        # Setup 3D plot
        if self.args.enable_3d_plot:
            plt.rcParams["keymap.quit"] = ["q", "escape"]
            self.fig, self.ax = plt.subplots(len(self.env.unwrapped.cameras), 1, subplot_kw=dict(projection="3d"))
            self.fig.tight_layout()
            self.point_cloud_scatter_list = [None] * len(self.env.unwrapped.cameras)

    def run(self):
        reset = True
        while True:
            iteration_start_time = time.time()

            # Reset
            if reset:
                self.motion_manager.reset()
                if self.args.replay_log is None:
                    self.record_manager.reset()
                    world_idx = None
                else:
                    self.record_manager.loadData(self.args.replay_log)
                    world_idx = self.record_manager.data_seq["world_idx"].tolist()
                self.record_manager.setupSimWorld(world_idx)
                obs, info = self.env.reset()
                print("== [{}] data_idx: {}, world_idx: {} ==".format(
                    self.demo_name, self.record_manager.data_idx, self.record_manager.world_idx))
                print("- Press the 'n' key to start automatic grasping.")
                reset = False

            # Read spacemouse
            if self.record_manager.status == RecordStatus.TELEOP:
                # Empirically, you can call read repeatedly to get the latest device status
                for i in range(10):
                    self.spacemouse_state = pyspacemouse.read()

            # Get action
            if self.args.replay_log is not None and \
               self.record_manager.status in (RecordStatus.TELEOP, RecordStatus.END):
                action = self.record_manager.getSingleData(RecordKey.ACTION, teleop_time_idx)
            else:
                # Set commands
                self.setArmCommand()
                self.setGripperCommand()

                # Solve IK
                self.motion_manager.drawMarkers()
                self.motion_manager.inverseKinematics()

                action = self.motion_manager.getAction()

            # Record data
            if self.record_manager.status == RecordStatus.TELEOP and self.args.replay_log is None:
                self.record_manager.appendSingleData(RecordKey.TIME, self.record_manager.status_elapsed_duration)
                self.record_manager.appendSingleData(RecordKey.JOINT_POS, self.motion_manager.getJointPos(obs))
                self.record_manager.appendSingleData(RecordKey.JOINT_VEL, self.motion_manager.getJointVel(obs))
                for camera_name, rgb_key, depth_key in zip(self.camera_names, self.rgb_keys, self.depth_keys):
                    self.record_manager.appendSingleData(rgb_key, info["rgb_images"][camera_name])
                    self.record_manager.appendSingleData(depth_key, info["depth_images"][camera_name])
                self.record_manager.appendSingleData(RecordKey.WRENCH, obs[16:])
                self.record_manager.appendSingleData(RecordKey.MEASURED_EEF, self.motion_manager.getMeasuredEef(obs))
                self.record_manager.appendSingleData(RecordKey.COMMAND_EEF, self.motion_manager.getCommandEef())
                self.record_manager.appendSingleData(RecordKey.ACTION, action)

            # Step environment
            obs, _, _, _, info = self.env.step(action)

            # Draw images
            status_image = self.record_manager.getStatusImage()
            rgb_images = []
            depth_images = []
            for camera_name in self.camera_names:
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
            if self.args.enable_3d_plot:
                dist_thre_list = (3.0, 3.0, 0.8) # [m]
                for camera_idx, (camera_name, depth_key) in enumerate(zip(self.camera_names, self.depth_keys)):
                    point_cloud_skip = 10
                    small_depth_image = info["depth_images"][camera_name][::point_cloud_skip, ::point_cloud_skip]
                    small_rgb_image = info["rgb_images"][camera_name][::point_cloud_skip, ::point_cloud_skip]
                    fovy = self.record_manager.camera_info[depth_key.key() + "_fovy"]
                    xyz_array, rgb_array = convertDepthImageToPointCloud(
                        small_depth_image, fovy=fovy, rgb_image=small_rgb_image, dist_thre=dist_thre_list[camera_idx])
                    if self.point_cloud_scatter_list[camera_idx] is None:
                        get_min_max = lambda v_min, v_max: (0.75 * v_min + 0.25 * v_max, 0.25 * v_min + 0.75 * v_max)
                        self.ax[camera_idx].view_init(elev=-90, azim=-90)
                        self.ax[camera_idx].set_xlim(*get_min_max(xyz_array[:, 0].min(), xyz_array[:, 0].max()))
                        self.ax[camera_idx].set_ylim(*get_min_max(xyz_array[:, 1].min(), xyz_array[:, 1].max()))
                        self.ax[camera_idx].set_zlim(*get_min_max(xyz_array[:, 2].min(), xyz_array[:, 2].max()))
                    else:
                        self.point_cloud_scatter_list[camera_idx].remove()
                    self.ax[camera_idx].axis("off")
                    self.ax[camera_idx].set_aspect("equal")
                    self.point_cloud_scatter_list[camera_idx] = self.ax[camera_idx].scatter(
                        xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], c=rgb_array)
                plt.draw()
                plt.pause(0.001)

            # Manage status
            if self.record_manager.status == RecordStatus.INITIAL:
                if key == ord("n"):
                    self.record_manager.goToNextStatus()
            elif self.record_manager.status == RecordStatus.PRE_REACH:
                pre_reach_duration = 0.7 # [s]
                if self.record_manager.status_elapsed_duration > pre_reach_duration:
                    self.record_manager.goToNextStatus()
            elif self.record_manager.status == RecordStatus.REACH:
                reach_duration = 0.3 # [s]
                if self.record_manager.status_elapsed_duration > reach_duration:
                    self.record_manager.goToNextStatus()
                    print("- Press the 'n' key to start teleoperation after the gripper is closed.")
            elif self.record_manager.status == RecordStatus.GRASP:
                if key == ord("n"):
                    # Setup spacemouse
                    pyspacemouse.open()
                    teleop_time_idx = 0
                    self.record_manager.goToNextStatus()
                    if self.args.replay_log is None:
                        print("- Press the 'n' key to finish teleoperation.")
                    else:
                        print("- Start to replay the log motion.")
            elif self.record_manager.status == RecordStatus.TELEOP:
                teleop_time_idx += 1
                if self.args.replay_log is None:
                    if key == ord("n"):
                        print("- Press the 's' key if the teleoperation succeeded,"
                              " or the 'f' key if it failed. (duration: {:.1f} [s])".format(
                            self.record_manager.status_elapsed_duration))
                        self.record_manager.goToNextStatus()
                else:
                    if teleop_time_idx == len(self.record_manager.data_seq["time"]):
                        teleop_time_idx -= 1
                        self.record_manager.goToNextStatus()
                        print("- The log motion has finished replaying. Press the 'n' key to exit.")
            elif self.record_manager.status == RecordStatus.END:
                if self.args.replay_log is None:
                    if key == ord("s"):
                        # Save data
                        filename = "teleop_data/{}/env{:0>1}/{}_env{:0>1}_{:0>3}.npz".format(
                            self.demo_name, self.record_manager.world_idx,
                            self.demo_name, self.record_manager.world_idx, self.record_manager.data_idx)
                        if self.args.compress_rgb:
                            print("- Compress rgb images")
                            for rgb_key in self.rgb_keys:
                                self.record_manager.compressData(rgb_key, "jpg")
                        if self.args.compress_depth:
                            print("- Compress depth images")
                            for depth_key in self.depth_keys:
                                self.record_manager.compressData(depth_key, "exr")
                        self.record_manager.saveData(filename)
                        print("- Teleoperation succeeded: Save the data as {}".format(filename))
                        reset = True
                    elif key == ord("f"):
                        print("- Teleoperation failed: Reset without saving")
                        reset = True
                else:
                    if key == ord("n"):
                        break
            if key == 27: # escape key
                break

            iteration_duration = time.time() - iteration_start_time
            if iteration_duration < self.env.unwrapped.dt:
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        # self.env.close()

    def setArmCommand(self):
        if self.record_manager.status == RecordStatus.TELEOP:
            pos_scale = 1e-2
            delta_pos = pos_scale * np.array([
                -1.0 * self.spacemouse_state.y, self.spacemouse_state.x, self.spacemouse_state.z])
            rpy_scale = 5e-3
            delta_rpy = rpy_scale * np.array([
                -1.0 * self.spacemouse_state.roll, -1.0 * self.spacemouse_state.pitch, -2.0 * self.spacemouse_state.yaw])
            self.motion_manager.setRelativeTargetSE3(delta_pos, delta_rpy)

    def setGripperCommand(self):
        if self.record_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.high[6]
        elif self.record_manager.status == RecordStatus.TELEOP:
            gripper_scale = 5.0
            if self.spacemouse_state.buttons[0] > 0 and self.spacemouse_state.buttons[1] <= 0:
                self.motion_manager.gripper_pos += gripper_scale
            elif self.spacemouse_state.buttons[1] > 0 and self.spacemouse_state.buttons[0] <= 0:
                self.motion_manager.gripper_pos -= gripper_scale
