import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import MotionStatus
from robo_manip_baselines.teleop import TeleopBase


class TeleopMujocoXarm7Cable(TeleopBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoXarm7CableEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoXarm7Cable"

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_body_pose("cable_end")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] = 1.0  # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] = 0.925  # [m]
            self.motion_manager.target_se3 = pin.SE3(
                pin.rpy.rpyToMatrix(np.pi, 0.0, -np.pi / 2), target_pos
            )
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopMujocoXarm7Cable()
    teleop.run()
