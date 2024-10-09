import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.common import MotionStatus
from .RolloutBase import RolloutBase

class RolloutIsaacUR5eChain(RolloutBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/IsaacUR5eChainEnv-v0",
            render_mode="human"
        )

    def setArmCommand(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_link_pose("chain_end", "box")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] += 0.22 # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] += 0.14 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
            self.motion_manager.inverseKinematics()
        else:
            super().setArmCommand()

    def setGripperCommand(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_pos = 150.0
        else:
            super().setGripperCommand()
