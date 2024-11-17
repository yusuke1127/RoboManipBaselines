import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from multimodal_robot_model.teleop import TeleopBase
from multimodal_robot_model.common import MotionStatus

class TeleopMujocoXarm7Cable(TeleopBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoXarm7CableEnv-v0",
            render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoXarm7Cable"

    def setArmCommand(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_body_pose("cable_end")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] = 1.0 # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] = 0.925 # [m]
            self.motion_manager.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi, 0.0, -np.pi/2), target_pos)
        else:
            super().setArmCommand()

if __name__ == "__main__":
    teleop = TeleopMujocoXarm7Cable()
    teleop.run()
