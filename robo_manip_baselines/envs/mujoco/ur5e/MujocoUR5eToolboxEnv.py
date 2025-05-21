from os import path

import mujoco
import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eToolboxEnv(MujocoUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_toolbox.xml",
            ),
            np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.55 * np.pi,
                    -0.45 * np.pi,
                    np.pi / 2,
                    np.pi,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        # self.original_toolbox_pos = self.model.body("toolbox").pos.copy()
        # self.toolbox_pos_offsets = np.array(
        #     [
        #         [0.0, -0.06, 0.0],
        #         [0.0, -0.03, 0.0],
        #         [0.0, 0.0, 0.0],
        #         [0.0, 0.03, 0.0],
        #         [0.0, 0.06, 0.0],
        #         [0.0, 0.09, 0.0],
        #     ]
        # )  # [m]

    def _get_success(self):
        toolbox_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "toolbox"
        )
        toolbox_pos = self.data.xpos[toolbox_body_id].copy()
        mat_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mat")
        mat_pos = self.data.xpos[mat_body_id].copy()

        xy_thre = 0.03  # [m]
        z_thre = mat_pos[2] + 0.005  # [m]
        return (np.max(np.abs(toolbox_pos[:2] - mat_pos[:2])) < xy_thre) and (
            toolbox_pos[2] < z_thre
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        # if world_idx is None:
        #     world_idx = cumulative_idx % len(self.toolbox_pos_offsets)
        world_idx = 0

        return world_idx
