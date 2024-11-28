import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.common import MotionStatus
from .RolloutBase import RolloutBase

class RolloutMujocoUR5eCloth(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eClothEnv-v0",
            render_mode="human"
        )

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, 0.25*np.pi),
                                 self.env.unwrapped.get_body_pose("board")[0:3])
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_se3 *= pin.SE3(pin.rpy.rpyToMatrix(0.0, 0.125*np.pi, 0.0), np.array([0.0, -0.2, -0.4]))
            elif self.data_manager.status == MotionStatus.REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.3]))
            self.motion_manager.target_se3 = target_se3
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()

    def set_gripper_command(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.low[self.env.unwrapped.gripper_action_idx]
        else:
            super().set_gripper_command()
