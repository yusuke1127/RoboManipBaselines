import numpy as np
import pinocchio as pin
import gymnasium as gym
from robo_manip_baselines.common import MotionStatus
from .RolloutBase import RolloutBase


class RolloutIsaacUR5eChain(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/IsaacUR5eChainEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            target_pos = self.env.unwrapped.get_link_pose("chain_end", "box")[0:3]
            if self.data_manager.status == MotionStatus.PRE_REACH:
                target_pos[2] += 0.22  # [m]
            elif self.data_manager.status == MotionStatus.REACH:
                target_pos[2] += 0.14  # [m]
            self.motion_manager.target_se3 = pin.SE3(
                np.diag([-1.0, 1.0, -1.0]), target_pos
            )
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()

    def set_gripper_command(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_joint_pos = np.array([150.0])
        else:
            super().set_gripper_command()
