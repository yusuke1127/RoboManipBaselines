import numpy as np
import gymnasium as gym
import pinocchio as pin
from multimodal_robot_model.teleop import TeleopBase
from multimodal_robot_model.common import MotionStatus


class TeleopMujocoUR5eParticle(TeleopBase):
    def __init__(self):
        super().__init__()

        # Command configuration
        self.command_rpy_scale = 1e-2

    def setup_env(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eParticleEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoUR5eParticle"

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
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopMujocoUR5eParticle()
    teleop.run()
