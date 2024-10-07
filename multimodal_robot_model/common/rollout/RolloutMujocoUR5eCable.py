import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.common import RecordStatus
from .RolloutBase import RolloutBase

class RolloutMujocoUR5eCable(RolloutBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eCableEnv-v0",
            render_mode="human"
        )

    def setCommand(self):
        # Set joint command
        if self.data_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            target_pos = self.env.unwrapped.get_body_pose("cable_end")[0:3]
            if self.data_manager.status == RecordStatus.PRE_REACH:
                target_pos[2] = 1.02 # [m]
            elif self.data_manager.status == RecordStatus.REACH:
                target_pos[2] = 0.995 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
            self.motion_manager.inverseKinematics()
        elif self.data_manager.status == RecordStatus.TELEOP:
            self.motion_manager.joint_pos = self.pred_action[:6]

        # Set gripper command
        if self.data_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.high[6]
        elif self.data_manager.status == RecordStatus.TELEOP:
            self.motion_manager.gripper_pos = self.pred_action[6]
