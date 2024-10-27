import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from multimodal_robot_model.teleop import TeleopBase
from multimodal_robot_model.common import MotionStatus

class TeleopMujocoAlohaCable(TeleopBase):
    def __init__(self):
        super().__init__()

        # Command configuration
        self.command_rpy_scale = 2e-2

    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoAlohaCableEnv-v0",
            render_mode="human"
        )
        self.demo_name = "MujocoAlohaCable"

    def setArmCommand(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_body_pose("B0")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] = 0.30 # [m]
                target_rpy = np.array([0.0, np.deg2rad(30), -np.pi/2])
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] = 0.25 # [m]
                target_rpy = np.array([0.0, np.deg2rad(60), -np.pi/2])
            self.motion_manager.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(*target_rpy), target_pos)
        else:
            super().setArmCommand()

if __name__ == "__main__":
    teleop = TeleopMujocoAlohaCable()
    teleop.run()
