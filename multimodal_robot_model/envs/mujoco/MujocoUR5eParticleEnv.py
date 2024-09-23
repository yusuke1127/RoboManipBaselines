from os import path
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase

class MujocoUR5eParticleEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        extra_camera_configs=None,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(path.dirname(__file__), "../assets/mujoco/envs/env_ur5e_particle.xml"),
            np.array([np.pi, -np.pi/2, -0.75*np.pi, -0.25*np.pi, np.pi/2, np.pi, 0.0]),
            extra_camera_configs,
            **kwargs)

        self.original_source_pos = self.model.body("source_case").pos.copy()
        self.original_particle_pos = self.model.body("particle").pos.copy()
        self.pos_offsets = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.12, 0.0],
            [0.0, 0.16, 0.0],
            [0.0, 0.20, 0.0],
        ]) # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pos_offsets)
        self.model.body("source_case").pos = self.original_source_pos + self.pos_offsets[world_idx]
        self.model.body("particle").pos = self.original_particle_pos + self.pos_offsets[world_idx]
        return world_idx
