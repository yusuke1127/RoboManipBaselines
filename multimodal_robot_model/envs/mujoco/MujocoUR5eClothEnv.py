from os import path
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase

class MujocoUR5eClothEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(path.dirname(__file__), "../assets/mujoco/envs/env_ur5e_cloth.xml"),
            np.array([np.pi, -np.pi/2, -0.75*np.pi, -0.75*np.pi, -0.5*np.pi, 0.0, 0.0]),
            extra_camera_configs=[
                {"name": "front", "size": (480, 640)},
                {"name": "side", "size": (480, 640)},
                {"name": "hand", "size": (480, 640)},
            ],
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
