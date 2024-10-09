import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.common import MotionStatus
from .RolloutBase import RolloutBase

class RolloutMujocoUR5eCloth(RolloutBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eClothEnv-v0",
            render_mode="human"
        )

    def setArmCommand(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, 0.25*np.pi),
                                 self.env.unwrapped.get_body_pose("cloth_root_frame")[0:3])
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.25]))
            elif self.data_manager.status == MotionStatus.REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.2]))
            self.motion_manager.target_se3 = target_se3
            self.motion_manager.inverseKinematics()
        else:
            super().setArmCommand()

    def setGripperCommand(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.low[6]
        else:
            super().setGripperCommand()
