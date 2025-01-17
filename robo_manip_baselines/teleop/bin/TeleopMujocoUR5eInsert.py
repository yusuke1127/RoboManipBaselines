import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import Phase
from robo_manip_baselines.teleop import TeleopBase


class TeleopMujocoUR5eInsert(TeleopBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eInsertEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eInsert"

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_pos = self.env.unwrapped.get_body_pose("peg")[0:3]
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_pos[2] = 1.1  # [m]
            elif self.phase_manager.phase == Phase.REACH:
                target_pos[2] = 1.03  # [m]
            self.motion_manager.target_se3 = pin.SE3(
                np.diag([-1.0, 1.0, -1.0]), target_pos
            )
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopMujocoUR5eInsert()
    teleop.run()
