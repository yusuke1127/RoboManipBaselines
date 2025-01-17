import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import Phase
from robo_manip_baselines.teleop import TeleopBase


class TeleopIsaacUR5eCabinet(TeleopBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eCabinetEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "IsaacUR5eCabinet"

    def set_arm_command(self):
        if self.phase_manager.phase in (Phase.PRE_REACH, Phase.REACH):
            target_pos = self.env.unwrapped.get_link_pose("ur5e", "base_link")[0:3]
            if self.phase_manager.phase == Phase.PRE_REACH:
                target_pos += np.array([0.33, 0.0, 0.3])  # [m]
            elif self.phase_manager.phase == Phase.REACH:
                target_pos += np.array([0.38, 0.0, 0.3])  # [m]
            self.motion_manager.target_se3.translation = target_pos
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopIsaacUR5eCabinet()
    teleop.run()
