from os import path
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "azimuth": -135.0,
    "elevation": -45.0,
    "distance": 2.0,
    "lookat": [-0.2, 0.0, 0.8]
}

class UR5eCableEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file=path.join(path.dirname(__file__), "assets/envs/env_ur5e_cable_verticalup.xml"),
        # xml_file=path.join(path.dirname(__file__), "assets/envs/env_ur5e_cable_diagonaldown.xml"),
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            **kwargs,
        )

        obs_shape = 22
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.init_qpos[:6] = np.array([np.pi, -np.pi/2, -0.75*np.pi, -0.25*np.pi, np.pi/2, np.pi/2]) # env_ur5e_cable_verticalup.xml
        # self.init_qpos[:6] = np.array([1.0472, -2.26893, 2.0944, -1.8326, -1.48353, -0.698132]) # env_ur5e_cable_diagonaldown.xml
        self.init_qvel[:] = 0.0

    @property
    def urdf_path(self):
        return path.join(path.dirname(__file__), "assets/robots/ur5e/ur5e.urdf")

    @property
    def terminated(self):
        # TODO
        return False

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward = 0.0
        terminated = self.terminated
        info = {
        }

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

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
        force = self.data.sensor("force_sensor").data.flat.copy()
        torque = self.data.sensor("torque_sensor").data.flat.copy()
        return np.concatenate((arm_qpos, arm_qvel, gripper_qpos, force, torque))

    def reset_model(self):
        reset_noise_scale = 0.0

        qpos = self.init_qpos + self.np_random.uniform(
            low=-1 * reset_noise_scale, high=reset_noise_scale, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
