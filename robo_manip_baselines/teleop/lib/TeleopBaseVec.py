import time

import numpy as np
import pyspacemouse

from robo_manip_baselines.common import DataKey, DataManagerVec, MotionStatus

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
                        world_idx = self.args.world_idx_list[
                            self.data_manager.episode_idx
                            % len(self.args.world_idx_list)
                        ]
                else:
                    raise NotImplementedError(
                        '[TeleopBaseVec] The "replay_log" option is not supported.'
                    )
                    self.data_manager.load_data(self.args.replay_log)
                    print(
                        "[TeleopBaseVec] Load teleoperation data: {}".format(
                            self.args.replay_log
                        )
                    )
                    world_idx = self.data_manager.get_data("world_idx").tolist()
                self.data_manager.setup_sim_world(world_idx)
                self.env.reset()
                obs_list = self.env.unwrapped.obs_list
                info_list = self.env.unwrapped.info_list
                print(
                    "[{}] episode_idx: {}, world_idx: {}".format(
                        self.demo_name,
                        self.data_manager.episode_idx,
                        self.data_manager.world_idx,
                    )
                )
                print("[TeleopBaseVec] Press the 'n' key to start automatic grasping.")
                self.reset_flag = False

            # Read spacemouse
            if self.data_manager.status == MotionStatus.TELEOP:
                # Empirically, you can call read repeatedly to get the latest device status
                for i in range(10):
                    self.spacemouse_state = pyspacemouse.read()

            # Get action
            if self.args.replay_log is not None and self.data_manager.status in (
                MotionStatus.TELEOP,
                MotionStatus.END,
            ):
                action = self.data_manager.get_single_data(
                    DataKey.COMMAND_JOINT_POS, self.teleop_time_idx
                )
            else:
                # Set commands
                self.set_arm_command()
                self.set_gripper_command()

                # Solve IK
                self.motion_manager.draw_markers()
                self.motion_manager.inverse_kinematics()

                # Set action
                action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
                update_fluctuation = self.data_manager.status == MotionStatus.TELEOP
                action_list = self.env.unwrapped.get_fluctuated_action_list(
                    action, update_fluctuation
                )

            # Record data
            if (
                self.data_manager.status == MotionStatus.TELEOP
                and self.args.replay_log is None
            ):
                # Add time
                self.data_manager.append_single_data(
                    DataKey.TIME,
                    [self.data_manager.status_elapsed_duration]
                    * self.env.unwrapped.num_envs,
                )

                # Add measured data
                for key in (
                    DataKey.MEASURED_JOINT_POS,
                    DataKey.MEASURED_JOINT_VEL,
                    DataKey.MEASURED_GRIPPER_JOINT_POS,
                    DataKey.MEASURED_EEF_POSE,
                    DataKey.MEASURED_EEF_WRENCH,
                ):
                    self.data_manager.append_single_data(
                        key,
                        [
                            self.motion_manager.get_measured_data(key, obs)
                            for obs in obs_list
                        ],
                    )

                # Add command data
                for key in (
                    DataKey.COMMAND_JOINT_POS,
                    DataKey.COMMAND_GRIPPER_JOINT_POS,
                    # TODO: COMMAND_EEF_POSE does not reflect the effect of action fluctuation
                    DataKey.COMMAND_EEF_POSE,
                ):
                    self.data_manager.append_single_data(
                        key,
                        [
                            self.motion_manager.get_command_data(key)
                            for action in action_list
                        ],
                    )

                # Add relative data
                for key in (
                    DataKey.MEASURED_JOINT_POS_REL,
                    DataKey.COMMAND_JOINT_POS_REL,
                    DataKey.MEASURED_EEF_POSE_REL,
                    DataKey.COMMAND_EEF_POSE_REL,
                ):
                    self.data_manager.append_single_data(
                        key, self.data_manager.calc_relative_data(key)
                    )

                # Add image
                for camera_name in self.env.unwrapped.camera_names:
                    self.data_manager.append_single_data(
                        DataKey.get_rgb_image_key(camera_name),
                        [info["rgb_images"][camera_name] for info in info_list],
                    )
                    self.data_manager.append_single_data(
                        DataKey.get_depth_image_key(camera_name),
                        [info["depth_images"][camera_name] for info in info_list],
                    )

            # Step environment
            self.env.unwrapped.action_list = action_list
            self.env.step(action)
            obs_list = self.env.unwrapped.obs_list
            info_list = self.env.unwrapped.info_list

            # Draw images
            self.draw_image(info_list[self.env.unwrapped.rep_env_idx])

            # Draw point clouds
            if self.args.enable_3d_plot:
                self.draw_point_cloud(info_list[[self.env.unwrapped.rep_env_idx]])

            # Manage status
            self.manage_status()
            if self.quit_flag:
                break

            iteration_duration = time.time() - iteration_start_time
            if self.data_manager.status == MotionStatus.TELEOP:
                iteration_duration_list.append(iteration_duration)
            if iteration_duration < self.env.unwrapped.dt:
                time.sleep(self.env.unwrapped.dt - iteration_duration)

        print("[TeleopBaseVec] Statistics on teleoperation")
        if len(iteration_duration_list) > 0:
            iteration_duration_list = np.array(iteration_duration_list)
            print(
                f"  - Real-time factor | {self.env.unwrapped.dt / iteration_duration_list.mean():.2f}"
            )
            print(
                "  - Iteration duration [s] | "
                f"mean: {iteration_duration_list.mean():.3f}, std: {iteration_duration_list.std():.3f} "
                f"min: {iteration_duration_list.min():.3f}, max: {iteration_duration_list.max():.3f}"
            )

        # self.env.close()

    def save_data(self):
        filename_list = []
        aug_idx = 0
        for env_idx, success in enumerate(self.env.unwrapped.success_list):
            if not success:
                filename_list.append(None)
                continue
            if env_idx == self.env.unwrapped.rep_env_idx:
                extra_label = "nominal"
            else:
                extra_label = f"augmented{aug_idx:0>3}"
                aug_idx += 1
            filename = "teleop_data/{}_{:%Y%m%d_%H%M%S}/env{:0>1}/{}_env{:0>1}_{:0>3}_{}.hdf5".format(
                self.demo_name,
                self.datetime_now,
                self.data_manager.world_idx,
                self.demo_name,
                self.data_manager.world_idx,
                self.data_manager.episode_idx,
                extra_label,
            )
            filename_list.append(filename)
        self.data_manager.save_data(filename_list)
        num_success = sum(filename is not None for filename in filename_list)
        if num_success > 0:
            print(
                "[TeleopBaseVec] Teleoperation succeeded: Save the {} data such as {}, etc.".format(
                    sum(filename is not None for filename in filename_list),
                    next(
                        filename for filename in filename_list if filename is not None
                    ),
                )
            )
        else:
            print(
                "[TeleopBaseVec] Teleoperation succeeded: Save no data because there is no successful data."
            )
