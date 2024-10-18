import time
import numpy as np
import pyspacemouse
from multimodal_robot_model.common import MotionStatus, DataKey, DataManagerVec
from .TeleopBase import TeleopBase

class TeleopBaseVec(TeleopBase):
    def __init__(self):
        self.DataManagerClass = DataManagerVec

        super().__init__()

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        iteration_duration_list = []

        while True:
            iteration_start_time = time.time()

            # Reset
            if self.reset_flag:
                self.motion_manager.reset()
                if self.args.replay_log is None:
                    self.data_manager.reset()
                    if self.args.world_idx_list is None:
                        world_idx = None
                    else:
                        world_idx = self.args.world_idx_list[self.data_manager.data_idx % len(self.args.world_idx_list)]
                else:
                    raise NotImplementedError("[TeleopBaseVec] The \"replay_log\" option is not supported.")
                    self.data_manager.loadData(self.args.replay_log)
                    print("- Load teleoperation data: {}".format(self.args.replay_log))
                    world_idx = self.data_manager.getData("world_idx").tolist()
                self.data_manager.setupSimWorld(world_idx)
                self.env.reset()
                obs_list, info_list = self.env.unwrapped.obs_list, self.env.unwrapped.info_list
                print("[{}] data_idx: {}, world_idx: {}".format(
                    self.demo_name, self.data_manager.data_idx, self.data_manager.world_idx))
                print("- Press the 'n' key to start automatic grasping.")
                self.reset_flag = False

            # Read spacemouse
            if self.data_manager.status == MotionStatus.TELEOP:
                # Empirically, you can call read repeatedly to get the latest device status
                for i in range(10):
                    self.spacemouse_state = pyspacemouse.read()

            # Get action
            if self.args.replay_log is not None and \
               self.data_manager.status in (MotionStatus.TELEOP, MotionStatus.END):
                action = self.data_manager.getSingleData(DataKey.COMMAND_JOINT_POS, self.teleop_time_idx)
            else:
                # Set commands
                self.setArmCommand()
                self.setGripperCommand()

                # Solve IK
                self.motion_manager.drawMarkers()
                self.motion_manager.inverseKinematics()

                action = self.motion_manager.getAction()
                update_fluctuation = (self.data_manager.status == MotionStatus.TELEOP)
                action_list = self.env.unwrapped.get_fluctuated_action_list(action, update_fluctuation)

            # Record data
            if self.data_manager.status == MotionStatus.TELEOP and self.args.replay_log is None:
                self.data_manager.appendSingleData(
                    DataKey.TIME,
                    [self.data_manager.status_elapsed_duration] * self.env.unwrapped.num_envs)
                self.data_manager.appendSingleData(
                    DataKey.MEASURED_JOINT_POS,
                    [self.motion_manager.getJointPos(obs) for obs in obs_list])
                self.data_manager.appendSingleData(
                    DataKey.COMMAND_JOINT_POS,
                    action_list)
                self.data_manager.appendSingleData(
                    DataKey.MEASURED_JOINT_VEL,
                    [self.motion_manager.getJointVel(obs) for obs in obs_list])
                self.data_manager.appendSingleData(
                    DataKey.MEASURED_EEF_POSE,
                    [self.motion_manager.getMeasuredEef(obs) for obs in obs_list])
                # TODO: COMMAND_EEF_POSE does not reflect the effect of action fluctuation
                self.data_manager.appendSingleData(
                    DataKey.COMMAND_EEF_POSE,
                    [self.motion_manager.getCommandEef()] * self.env.unwrapped.num_envs)
                self.data_manager.appendSingleData(
                    DataKey.MEASURED_WRENCH,
                    [self.motion_manager.getEefWrench(obs) for obs in obs_list])
                for camera_name in self.env.unwrapped.camera_names:
                    self.data_manager.appendSingleData(
                        DataKey.getRgbImageKey(camera_name),
                        [info["rgb_images"][camera_name] for info in info_list])
                    self.data_manager.appendSingleData(
                        DataKey.getDepthImageKey(camera_name),
                        [info["depth_images"][camera_name] for info in info_list])

            # Step environment
            self.env.unwrapped.action_list = action_list
            self.env.step(action)
            obs_list, info_list = self.env.unwrapped.obs_list, self.env.unwrapped.info_list

            # Draw images
            self.drawImage(info_list[self.env.unwrapped.rep_env_idx])

            # Draw point clouds
            if self.args.enable_3d_plot:
                self.drawPointCloud(info_list[[self.env.unwrapped.rep_env_idx]])

            # Manage status
            self.manageStatus()
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.data_manager.status == MotionStatus.TELEOP:
                iteration_duration_list.append(iteration_duration)
            if iteration_duration < self.env.unwrapped.dt:
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        print("- Statistics on teleoperation")
        if len(iteration_duration_list) > 0:
            iteration_duration_list = np.array(iteration_duration_list)
            print(f"  - Real-time factor | {self.env.unwrapped.dt / iteration_duration_list.mean():.2f}")
            print("  - Iteration duration [s] | "
                  f"mean: {iteration_duration_list.mean():.3f}, std: {iteration_duration_list.std():.3f} "
                  f"min: {iteration_duration_list.min():.3f}, max: {iteration_duration_list.max():.3f}")

        # self.env.close()

    def saveData(self):
        filename_list = []
        for env_idx in range(self.env.unwrapped.num_envs):
            filename = "teleop_data/{}/env{:0>1}/{}_env{:0>1}_{:0>3}_{:0>3}.npz".format(
                self.demo_name, self.data_manager.world_idx,
                self.demo_name, self.data_manager.world_idx, self.data_manager.data_idx,
                env_idx)
            filename_list.append(filename)
        if self.args.compress_rgb:
            print("- Compress rgb images")
            for camera_name in self.env.unwrapped.camera_names:
                self.data_manager.compressData(DataKey.getRgbImageKey(camera_name), "jpg")
        if self.args.compress_depth:
            print("- Compress depth images")
            for camera_name in self.env.unwrapped.camera_names:
                self.data_manager.compressData(DataKey.getDepthImageKey(camera_name), "exr")
        self.data_manager.saveData(filename_list)
        print("- Teleoperation succeeded: Save the data as {}, etc.".format(filename_list[0]))
