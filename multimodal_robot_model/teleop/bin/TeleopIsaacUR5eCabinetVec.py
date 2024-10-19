import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from multimodal_robot_model.teleop import TeleopBaseVec
from multimodal_robot_model.common import MotionStatus

class TeleopIsaacUR5eCabinetVec(TeleopBaseVec):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/IsaacUR5eCabinetEnv-v0",
            num_envs=12,
            render_mode="human"
        )
        self.demo_name = "IsaacUR5eCabinetVec"

    def setArmCommand(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_link_pose("ur5e", "base_link")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos += np.array([0.33, 0.0, 0.3]) # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos += np.array([0.33, 0.0, 0.3]) # [m]
            self.motion_manager.target_se3.translation = target_pos
        else:
            super().setArmCommand()

if __name__ == "__main__":
    teleop = TeleopIsaacUR5eCabinetVec()
    teleop.run()
