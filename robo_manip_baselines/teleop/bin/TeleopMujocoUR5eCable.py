import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import MotionStatus
from robo_manip_baselines.teleop import TeleopBase


class TeleopMujocoUR5eCable(TeleopBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoUR5eCableEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eCable"

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_body_pose("cable_end")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] = 1.02  # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] = 0.995  # [m]
            self.motion_manager.target_se3 = pin.SE3(
                np.diag([-1.0, 1.0, -1.0]), target_pos
            )
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopMujocoUR5eCable()
    teleop.run()
