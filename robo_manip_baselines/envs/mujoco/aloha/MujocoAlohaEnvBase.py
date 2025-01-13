from os import path
import numpy as np
import mujoco
from gymnasium.spaces import Box, Dict

from ..MujocoEnvBase import MujocoEnvBase


class MujocoAlohaEnvBase(MujocoEnvBase):
    default_camera_config = {
        "azimuth": 0.0,
        "elevation": -20.0,
        "distance": 1.8,
        "lookat": [0.0, 0.0, 0.3],
    }
    observation_space = Dict(
        {
            "left/joint_pos": Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            ),
            "left/joint_vel": Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            ),
            "right/joint_pos": Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            ),
            "right/joint_vel": Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            ),
        }
    )

    def setup_robot(self, init_qpos):
        mujoco.mj_kinematics(self.model, self.data)
        self.arm_urdf_path = path.join(
            path.dirname(__file__), "../../assets/common/robots/aloha/vx300s.urdf"
        )
        self.arm_root_pose = self.get_body_pose("left/base_link")
        self.ik_eef_joint_id = 6
        self.init_qpos[0 : len(init_qpos)] = init_qpos
        self.init_qpos[len(init_qpos) : 2 * len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        self.gripper_joint_idxes = [6]
        self.arm_joint_idxes = slice(0, 6)

    def step(self, action):
        # Copy the same action to both arms
        action = np.concatenate((action, action))
        return super().step(action)

    def _get_obs(self):
        obs = {
            "left/joint_pos": np.zeros(7),
            "left/joint_vel": np.zeros(7),
            "right/joint_pos": np.zeros(7),
            "right/joint_vel": np.zeros(7),
        }

        single_arm_joint_name_list = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]
        single_gripper_joint_name_list = [
            "left_finger",
            "right_finger",
        ]

        for arm_idx, arm_name in enumerate(("left", "right")):
            joint_pos_key = f"{arm_name}/joint_pos"
            joint_vel_key = f"{arm_name}/joint_vel"

            for joint_idx, joint_name in enumerate(single_arm_joint_name_list):
                joint_name = f"{arm_name}/{joint_name}"
                obs[joint_pos_key][joint_idx] = self.data.joint(joint_name).qpos[0]
                obs[joint_vel_key][joint_idx] = self.data.joint(joint_name).qvel[0]

            gripper_joint_qpos = []
            gripper_joint_qvel = []
            for joint_name in single_gripper_joint_name_list:
                joint_name = f"{arm_name}/{joint_name}"
                gripper_joint_qpos.append(self.data.joint(joint_name).qpos[0])
                gripper_joint_qvel.append(self.data.joint(joint_name).qvel[0])
            obs[joint_pos_key][-1] = np.array(gripper_joint_qpos).mean()
            obs[joint_vel_key][-1] = np.array(gripper_joint_qvel).mean()

        return obs

    def get_joint_pos_from_obs(self, obs, exclude_gripper=False):
        """Get joint position from observation."""
        if exclude_gripper:
            return obs["left/joint_pos"][self.arm_joint_idxes]
        else:
            return obs["left/joint_pos"]

    def get_joint_vel_from_obs(self, obs, exclude_gripper=False):
        """Get joint velocity from observation."""
        if exclude_gripper:
            return obs["left/joint_vel"][self.arm_joint_idxes]
        else:
            return obs["left/joint_vel"]

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return np.zeros(6)
