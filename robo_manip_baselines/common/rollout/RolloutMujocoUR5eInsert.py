import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import DataKey, Phase

from .RolloutBase import RolloutBase


class RolloutMujocoUR5eInsert(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eInsertEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_pos = self.env.unwrapped.get_body_pose("peg")[0:3]
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_pos[2] = 1.1  # [m]
            elif self.phase_manager.phase == Phase.REACH:
                target_pos[2] = 1.03  # [m]
            target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
            self.motion_manager.set_command_data(DataKey.COMMAND_EEF_POSE, target_se3)
        else:
            super().set_arm_command()
