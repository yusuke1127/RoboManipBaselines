import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import Phase
from robo_manip_baselines.teleop import TeleopBase


class TeleopMujocoUR5eParticle(TeleopBase):
    def __init__(self):
        super().__init__()

        # Command configuration
        self.command_rpy_scale = 1e-2

    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eParticleEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eParticle"

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_pos = self.env.unwrapped.get_geom_pose("scoop_handle")[0:3]
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_pos += np.array([0.0, 0.0, 0.2])  # [m]
            elif self.phase_manager.phase == Phase.REACH:
                target_pos += np.array([0.0, 0.0, 0.15])  # [m]
            self.motion_manager.target_se3 = pin.SE3(
                pin.rpy.rpyToMatrix(np.pi, 0.0, np.pi / 2), target_pos
            )
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopMujocoUR5eParticle()
    teleop.run()
