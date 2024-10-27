from os import path
import numpy as np
import mujoco
from gymnasium.spaces import Box, Dict

from ..MujocoEnvBase import MujocoEnvBase

class MujocoUR5eEnvBase(MujocoEnvBase):
    default_camera_config = {
        "azimuth": -135.0,
        "elevation": -45.0,
        "distance": 1.8,
        "lookat": [-0.2, -0.2, 0.8]
    }
    observation_space = Dict({
        "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
    })

    def setup_robot(self, init_qpos):
        mujoco.mj_kinematics(self.model, self.data)
        self.arm_urdf_path = path.join(path.dirname(__file__), "../../assets/common/robots/ur5e/ur5e.urdf")
        self.arm_root_pose = self.get_body_pose("ur5e_root_frame")
        self.init_qpos[:len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

    def _get_obs(self):
        arm_joint_name_list = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        gripper_joint_name_list = [
            "right_driver_joint",
            "right_spring_link_joint",
            "left_driver_joint",
            "left_spring_link_joint",
        ]

        arm_qpos = np.array([self.data.joint(joint_name).qpos[0] for joint_name in arm_joint_name_list])
        arm_qvel = np.array([self.data.joint(joint_name).qvel[0] for joint_name in arm_joint_name_list])
        gripper_qpos = np.array([self.data.joint(joint_name).qpos[0] for joint_name in gripper_joint_name_list])
        gripper_pos = np.rad2deg(gripper_qpos.mean(keepdims=True)) / 45.0 * 255.0
        force = self.data.sensor("force_sensor").data.flat.copy()
        torque = self.data.sensor("torque_sensor").data.flat.copy()

        return {
            "joint_pos": np.concatenate((arm_qpos, gripper_pos), dtype=np.float64),
            "joint_vel": np.concatenate((arm_qvel, np.zeros(1)), dtype=np.float64),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }

    def get_arm_qpos_from_obs(self, obs):
        """Grm arm joint position (6D array) from observation."""
        return obs["joint_pos"][:6]

    def get_arm_qvel_from_obs(self, obs):
        """Grm arm joint velocity (6D array) from observation."""
        return obs["joint_vel"][:6]

    def get_gripper_pos_from_obs(self, obs):
        """Grm gripper joint position (1D array) from observation."""
        return obs["joint_pos"][[6]]

    def get_eef_wrench_from_obs(self, obs):
        """Grm end-effector wrench (6D array) from observation."""
        return obs["wrench"]
