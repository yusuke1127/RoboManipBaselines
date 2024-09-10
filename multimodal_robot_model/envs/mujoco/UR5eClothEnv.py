from os import path
import numpy as np

from .UR5eEnvBase import UR5eEnvBase

class UR5eClothEnv(UR5eEnvBase):
    def __init__(
        self,
        extra_camera_configs=None,
        **kwargs,
    ):
        UR5eEnvBase.__init__(
            self,
            path.join(path.dirname(__file__), "assets/envs/env_ur5e_cloth.xml"),
            np.array([np.pi, -np.pi/2, -0.75*np.pi, -0.75*np.pi, -0.5*np.pi, 0.0]),
            extra_camera_configs,
            **kwargs)

        self.original_cloth_pos = self.model.body("cloth_root_frame").pos.copy()
        self.pos_cloth_offsets = np.array([
            [0.0, -0.08, 0.0],
            [0.0, -0.04, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.08, 0.0],
            [0.0, 0.12, 0.0],
        ]) # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pos_cloth_offsets)
        self.model.body("cloth_root_frame").pos = self.original_cloth_pos + self.pos_cloth_offsets[world_idx]
        return world_idx
