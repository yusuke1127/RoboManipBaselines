from os import path
import time
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict

import rtde_control
import rtde_receive
from gello.robots.robotiq_gripper import RobotiqGripper
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids

class RealUR5eEnvBase(gym.Env):
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
        self.dt = 0.02 # [s]
        if kwargs.get("scale_dt") is not None:
            self.dt *= kwargs["scale_dt"]
        self.action_space = Box(
            low=np.array([-2*np.pi, -2*np.pi, -1*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, 0.0], dtype=np.float32),
            high=np.array([2*np.pi, 2*np.pi, 1*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 255.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = Dict({
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        })

        # Setup robot
        self.arm_urdf_path = path.join(path.dirname(__file__), "../assets/common/robots/ur5e/ur5e.urdf")
        self.arm_root_pose = None
        self.init_qpos = init_qpos
        self.qvel_limit = np.deg2rad(191) # [rad/s]

        # Connect to UR5e
        print("[RealUR5eEnvBase] Start connecting the UR5e.")
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c.endFreedriveMode()
        self.arm_qpos_actual = np.array(self.rtde_r.getActualQ())
        print("[RealUR5eEnvBase] Finish connecting the UR5e.")

        # Connect to Robotiq gripper
        print("[RealUR5eEnvBase] Start connecting the Robotiq gripper.")
        self.gripper_port = 63352
        self.gripper = RobotiqGripper()
        self.gripper.connect(hostname=self.robot_ip, port=self.gripper_port)
        self._gripper_activated = False
        print("[RealUR5eEnvBase] Finish connecting the Robotiq gripper.")

        # Connect to RealSense
        self.cameras = {}
        detected_camera_ids = get_device_ids()
        for camera_name, camera_id in camera_ids.items():
            if camera_id is None:
                self.cameras[camera_name] = None
                continue

            if camera_id not in detected_camera_ids:
                raise ValueError(
                    f"Specified camera (name: {camera_name}, ID: {camera_id}) not detected. Detected camera IDs: {detected_camera_ids}")

            camera = RealSenseCamera(device_id=camera_id, flip=False)
            frames = camera._pipeline.wait_for_frames()
            color_intrinsics = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            camera.color_fovy = np.rad2deg(2 * np.arctan(color_intrinsics.height / (2 * color_intrinsics.fy)))
            depth_intrinsics = frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
            camera.depth_fovy = np.rad2deg(2 * np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy)))

            self.cameras[camera_name] = camera

    def reset(self, *, seed=None, options=None):
        self.init_time = time.time()

        super().reset(seed=seed)

        print("[RealUR5eEnvBase] Start moving the robot to the reset position.")
        self._set_action(self.init_qpos, duration=None, qvel_limit_scale=0.3, wait=True)
        print("[RealUR5eEnvBase] Finish moving the robot to the reset position.")

        if not self._gripper_activated:
            self._gripper_activated = True
            print("[RealUR5eEnvBase] Start activating the Robotiq gripper.")
            self.gripper.activate()
            print("[RealUR5eEnvBase] Finish activating the Robotiq gripper.")

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._set_action(action, duration=self.dt, qvel_limit_scale=2.0, wait=True)

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def close(self):
        pass

    def _set_action(self, action, duration=None, qvel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or qpos for safety
        arm_qpos_command = action[0:6]
        scaled_qvel_limit = np.clip(qvel_limit_scale, 0.01, 10.0) * self.qvel_limit
        if duration is None:
            duration_min, duration_max = 0.1, 10.0 # [s]
            duration = np.clip(np.max(np.abs(arm_qpos_command - self.arm_qpos_actual) / scaled_qvel_limit),
                               duration_min,
                               duration_max)
        else:
            arm_qpos_command_overwritten = self.arm_qpos_actual + np.clip(
                arm_qpos_command - self.arm_qpos_actual,
                -1 * scaled_qvel_limit * duration,
                scaled_qvel_limit * duration)
            # if np.linalg.norm(arm_qpos_command_overwritten - arm_qpos_command) > 1e-10:
            #     print("[RealUR5eEnvBase] Overwrite joint command for safety.")
            arm_qpos_command = arm_qpos_command_overwritten

        # Send command to UR5e
        velocity = 0.5
        acceleration = 0.5
        lookahead_time = 0.2 # [s]
        gain = 100
        period = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(arm_qpos_command, velocity, acceleration, duration, lookahead_time, gain)
        self.rtde_c.waitPeriod(period)

        # Send command to Robotiq gripper
        gripper_pos = action[6]
        speed = 50
        force = 10
        self.gripper.move(int(gripper_pos), speed, force)

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from UR5e
        arm_qpos = np.array(self.rtde_r.getActualQ())
        arm_qvel = np.array(self.rtde_r.getActualQd())
        self.arm_qpos_actual = arm_qpos.copy()

        # Get state from Robotiq gripper
        gripper_pos = np.array([self.gripper.get_current_position()], dtype=np.float64)

        # Get wrench from force sensor
        # Set zero because UR5e does not have a wrist force sensor
        force = np.zeros(3)
        torque = np.zeros(3)

        return {
            "joint_pos": np.concatenate((arm_qpos, gripper_pos), dtype=np.float64),
            "joint_vel": np.concatenate((arm_qvel, np.zeros(1)), dtype=np.float64),
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
                info["rgb_images"][camera_name] = np.zeros((480, 640, 3), dtype=np.uint8)
                info["depth_images"][camera_name] = np.zeros((480, 640), dtype=np.float32)
                continue

            rgb_image, depth_image = camera.read((640, 480))
            info["rgb_images"][camera_name] = rgb_image
            info["depth_images"][camera_name] = 1e-3 * depth_image[:, :, 0] # [m]

        return info

    def get_joint_pos_from_obs(self, obs, exclude_gripper=False):
        """Get joint position from observation."""
        if exclude_gripper:
            return obs["joint_pos"][:6]
        else:
            return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs, exclude_gripper=False):
        """Get joint velocity from observation."""
        if exclude_gripper:
            return obs["joint_vel"][:6]
        else:
            return obs["joint_vel"]

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (6D array) from observation."""
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
            return 45.0 # dummy
        return camera.depth_fovy

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError("[RealUR5eEnvBase] modify_world is not implemented.")

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # In a real-world environment, it is not possible to programmatically draw markers
        pass
