from os import path

import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eInsertEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_insert.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.65 * np.pi,
                    -0.35 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    0.0,
                ]
            ),
            **kwargs,
        )

        self.original_hole_pos = self.model.body("hole").pos.copy()
        self.hole_pos_offsets = np.array(
            [
                [-0.06, 0.0, 0.0],
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.09, 0.0, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.hole_pos_offsets)
        self.model.body("hole").pos = (
            self.original_hole_pos + self.hole_pos_offsets[world_idx]
        )
        return world_idx
