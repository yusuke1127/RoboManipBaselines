import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.common import MotionStatus
from .RolloutBase import RolloutBase


class RolloutMujocoXarm7Ring(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoXarm7RingEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = 0.5 * (
                self.env.unwrapped.get_geom_pose("fook1")[0:3]
                + self.env.unwrapped.get_geom_pose("fook2")[0:3]
            )
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos += np.array([-0.2, 0.05, -0.05])  # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos += np.array([-0.15, 0.05, -0.05])  # [m]
            self.motion_manager.target_se3 = pin.SE3(
                pin.rpy.rpyToMatrix(0.0, 1.5 * np.pi, np.pi), target_pos
            )
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()
