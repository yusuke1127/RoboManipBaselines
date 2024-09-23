from os import path
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase

class MujocoUR5eCableEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        extra_camera_configs=None,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(path.dirname(__file__), "../assets/mujoco/envs/env_ur5e_cable_verticalup.xml"),
            np.array([np.pi, -np.pi/2, -0.75*np.pi, -0.25*np.pi, np.pi/2, np.pi/2, 0.0]), # env_ur5e_cable_verticalup.xml
            # path.join(path.dirname(__file__), "../assets/mujoco/envs/env_ur5e_cable_diagonaldown.xml"),
            # np.array([1.0472, -2.26893, 2.0944, -1.8326, -1.48353, -0.698132, 0.0]), # env_ur5e_cable_diagonaldown.xml
            extra_camera_configs,
            **kwargs)

        self.original_pole_pos = self.model.body("poles").pos.copy()
        self.pole_pos_offsets = np.array([
            [-0.03, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.09, 0.0, 0.0],
            [0.12, 0.0, 0.0],
        ]) # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)
        self.model.body("poles").pos = self.original_pole_pos + self.pole_pos_offsets[world_idx]
        return world_idx
