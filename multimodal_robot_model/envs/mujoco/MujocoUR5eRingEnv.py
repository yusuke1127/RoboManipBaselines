from os import path
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase

class MujocoUR5eRingEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(path.dirname(__file__), "../assets/mujoco/envs/ur5e/env_ur5e_ring.xml"),
            np.array([np.pi, -np.pi/2, -0.75*np.pi, -0.75*np.pi, -0.5*np.pi, 0.0, 0.0]),
            **kwargs)

        self.original_pole_pos = self.model.body("pole").pos.copy()
        self.pole_pos_offsets = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.12, 0.0],
            [0.0, 0.16, 0.0],
            [0.0, 0.20, 0.0],
        ]) # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)
        self.model.body("pole").pos = self.original_pole_pos + self.pole_pos_offsets[world_idx]
        return world_idx
