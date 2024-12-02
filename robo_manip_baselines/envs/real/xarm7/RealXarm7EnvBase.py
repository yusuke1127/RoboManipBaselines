from os import path
import time
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict

from xarm.wrapper import XArmAPI
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids


class RealXarm7EnvBase(gym.Env):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        robot_ip,
        camera_ids,
        init_qpos,
        **kwargs,
    ):
        # Setup environment parameters
        self.init_time = time.time()
        self.dt = 0.02  # [s]
        if kwargs.get("scale_dt") is not None:
            self.dt *= kwargs["scale_dt"]

        self.action_space = Box(
            low=np.deg2rad(
                [
                    -2 * np.pi,
                    np.deg2rad(-118),
                    -2 * np.pi,
                    np.deg2rad(-11),
                    -2 * np.pi,
                    np.deg2rad(-97),
                    -2 * np.pi,
                    0.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    2 * np.pi,
                    np.deg2rad(120),
                    2 * np.pi,
                    np.deg2rad(225),
                    2 * np.pi,
                    np.pi,
                    2 * np.pi,
                    840.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.observation_space = Dict(
            {
                "joint_pos": Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
                ),
                "joint_vel": Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
                ),
                "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            }
        )

        self.gripper_action_idx = 7
        self.arm_action_idxes = slice(0, 7)

        # Setup robot
        self.arm_urdf_path = path.join(
            path.dirname(__file__), "../../assets/common/robots/xarm7/xarm7.urdf"
        )
        self.arm_root_pose = None
        self.ik_eef_joint_id = 7
        self.ik_arm_joint_ids = slice(0, 7)
        self.init_qpos = init_qpos
        self.qvel_limit = np.deg2rad(180)  # [rad/s]

        # Connect to xArm7
        print("[RealXarm7EnvBase] Start connecting the xArm7.")
        self.robot_ip = robot_ip
        self.xarm_api = XArmAPI(self.robot_ip)
        self.xarm_api.connect()
        self.xarm_api.motion_enable(enable=True)
        self.xarm_api.set_mode(0)
        self.xarm_api.set_state(0)
        self.xarm_api.clean_gripper_error()
        self.xarm_api.set_gripper_mode(0)
        self.xarm_api.set_gripper_enable(True)
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        self.arm_qpos_actual = joint_states[0]
        print("[RealXarm7EnvBase] Finish connecting the xArm7.")

        # Connect to RealSense
        self.cameras = {}
        detected_camera_ids = get_device_ids()
        for camera_name, camera_id in camera_ids.items():
            if camera_id is None:
                self.cameras[camera_name] = None
                continue

            if camera_id not in detected_camera_ids:
                raise ValueError(
                    f"Specified camera (name: {camera_name}, ID: {camera_id}) not detected. Detected camera IDs: {detected_camera_ids}"
                )

            camera = RealSenseCamera(device_id=camera_id, flip=False)
            frames = camera._pipeline.wait_for_frames()
            color_intrinsics = (
                frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.color_fovy = np.rad2deg(
                2 * np.arctan(color_intrinsics.height / (2 * color_intrinsics.fy))
            )
            depth_intrinsics = (
                frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.depth_fovy = np.rad2deg(
                2 * np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy))
            )

            self.cameras[camera_name] = camera

    def reset(self, *, seed=None, options=None):
        self.init_time = time.time()

        super().reset(seed=seed)

        print("[RealXarm7EnvBase] Start moving the robot to the reset position.")
        self._set_action(self.init_qpos, duration=None, qvel_limit_scale=0.1, wait=True)
        print("[RealXarm7EnvBase] Finish moving the robot to the reset position.")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._set_action(action, duration=self.dt, qvel_limit_scale=1.0, wait=True)

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def close(self):
        pass

    def _set_action(self, action, duration=None, qvel_limit_scale=0.5, wait=False):
        # start_time = time.time()

        # Overwrite duration or qpos for safety
        arm_qpos_command = action[self.arm_action_idxes]
        scaled_qvel_limit = np.clip(qvel_limit_scale, 0.01, 10.0) * self.qvel_limit
        if duration is None:
            duration_min, duration_max = 0.1, 10.0  # [s]
            duration = np.clip(
                np.max(
                    np.abs(arm_qpos_command - self.arm_qpos_actual) / scaled_qvel_limit
                ),
                duration_min,
                duration_max,
            )
        else:
            arm_qpos_command_overwritten = self.arm_qpos_actual + np.clip(
                arm_qpos_command - self.arm_qpos_actual,
                -1 * scaled_qvel_limit * duration,
                scaled_qvel_limit * duration,
            )
            # if np.linalg.norm(arm_qpos_command_overwritten - arm_qpos_command) > 1e-10:
            #     print("[RealXarm7EnvBase] Overwrite joint command for safety.")
            arm_qpos_command = arm_qpos_command_overwritten

        # Send command to xArm7
        xarm_code = self.xarm_api.set_servo_angle(
            angle=arm_qpos_command, speed=scaled_qvel_limit, is_radian=True, wait=False
        )
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")

        # Send command to xArm gripper
        gripper_pos = action[self.gripper_action_idx]
        xarm_code = self.xarm_api.set_gripper_position(gripper_pos)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")

        # Wait
        # elapsed_duration = time.time() - start_time
        # if wait and elapsed_duration < duration:
        #     time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from xArm7
        xarm_code, joint_states = self.xarm_api.get_joint_states(is_radian=True)
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        arm_qpos = joint_states[0]
        arm_qvel = joint_states[1]
        self.arm_qpos_actual = arm_qpos.copy()

        # Get state from Robotiq gripper
        xarm_code, gripper_pos = self.xarm_api.get_gripper_position()
        if xarm_code != 0:
            raise RuntimeError(f"[RealXarm7EnvBase] Invalid xArm API code: {xarm_code}")
        gripper_pos = np.array([gripper_pos], dtype=np.float64)
        gripper_vel = np.zeros(1)

        # Get wrench from force sensor
        wrench = np.array(self.xarm_api.get_ft_sensor_data()[1], dtype=np.float64)
        force = wrench[0:3]
        torque = wrench[3:6]

        return {
            "joint_pos": np.concatenate((arm_qpos, gripper_pos), dtype=np.float64),
            "joint_vel": np.concatenate((arm_qvel, gripper_vel), dtype=np.float64),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }

    def _get_info(self):
        info = {}

        if len(self.camera_names) == 0:
            return info

        # Get camera images
        info["rgb_images"] = {}
        info["depth_images"] = {}
        for camera_name, camera in self.cameras.items():
            if camera is None:
                info["rgb_images"][camera_name] = np.zeros(
                    (480, 640, 3), dtype=np.uint8
                )
                info["depth_images"][camera_name] = np.zeros(
                    (480, 640), dtype=np.float32
                )
                continue

            rgb_image, depth_image = camera.read((640, 480))
            info["rgb_images"][camera_name] = rgb_image
            info["depth_images"][camera_name] = 1e-3 * depth_image[:, :, 0]  # [m]

        return info

    def get_joint_pos_from_obs(self, obs, exclude_gripper=False):
        """Get joint position from observation."""
        if exclude_gripper:
            return obs["joint_pos"][self.arm_action_idxes]
        else:
            return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs, exclude_gripper=False):
        """Get joint velocity from observation."""
        if exclude_gripper:
            return obs["joint_vel"][self.arm_action_idxes]
        else:
            return obs["joint_vel"]

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return obs["wrench"]

    def get_sim_time(self):
        """Get simulation time. [s]"""
        return time.time() - self.init_time

    @property
    def camera_names(self):
        """Camera names being measured."""
        return self.cameras.keys()

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        camera = self.cameras[camera_name]
        if camera is None:
            return 45.0  # dummy
        return camera.depth_fovy

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError("[RealXarm7EnvBase] modify_world is not implemented.")

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # In a real-world environment, it is not possible to programmatically draw markers
        pass
