from os import path
import numpy as np

from .MujocoAlohaEnvBase import MujocoAlohaEnvBase

class MujocoAlohaCableEnv(MujocoAlohaEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoAlohaEnvBase.__init__(
            self,
            path.join(path.dirname(__file__), "../../assets/mujoco/envs/aloha/env_aloha_cable.xml"),
            np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.0084, 0.0084]),
            **kwargs)

    def modify_world(self, world_idx=None, cumulative_idx=None):
        # TODO: Modify the world settings according to world_idx
        if world_idx is None:
            world_idx = 0
        return world_idx
