import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.common import MotionStatus
from .RolloutBase import RolloutBase

class RolloutMujocoUR5eCable(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eCableEnv-v0",
            render_mode="human"
        )

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_body_pose("cable_end")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] = 1.02 # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] = 0.995 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()
