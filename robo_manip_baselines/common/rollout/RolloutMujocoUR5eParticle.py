import numpy as np
import pinocchio as pin
import gymnasium as gym
from robo_manip_baselines.common import MotionStatus
from .RolloutBase import RolloutBase


class RolloutMujocoUR5eParticle(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eParticleEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_geom_pose("scoop_handle")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos += np.array([0.0, 0.0, 0.2])  # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos += np.array([0.0, 0.0, 0.15])  # [m]
            self.motion_manager.target_se3 = pin.SE3(
                pin.rpy.rpyToMatrix(np.pi, 0.0, np.pi / 2), target_pos
            )
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()
