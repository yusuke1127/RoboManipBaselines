import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from multimodal_robot_model.teleop import TeleopBase
from multimodal_robot_model.common import MotionStatus

class TeleopMujocoUR5eCable(TeleopBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eCableEnv-v0",
            render_mode="human"
        )
        self.demo_name = "MujocoUR5eCable"

    def setArmCommand(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_body_pose("cable_end")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] = 1.02 # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] = 0.995 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
        else:
            super().setArmCommand()

if __name__ == "__main__":
    teleop = TeleopMujocoUR5eCable()
    teleop.run()
