import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.common import MotionManager, RecordStatus, RecordManager
from ..RolloutBase import RolloutBase

class RolloutMujocoUR5eCloth(RolloutBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eClothEnv-v0",
            render_mode="human",
            extra_camera_configs=[
                {"name": "front", "size": (480, 640)},
                {"name": "side", "size": (480, 640)},
                {"name": "hand", "size": (480, 640)},
            ]
        )

    def setCommand(self):
        # Set joint command
        if self.record_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, 0.25*np.pi),
                                 self.env.unwrapped.model.body("cloth_root_frame").pos.copy())
            if self.record_manager.status == RecordStatus.PRE_REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.25]))
            elif self.record_manager.status == RecordStatus.REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.2]))
            self.motion_manager.target_se3 = target_se3
            self.motion_manager.inverseKinematics()
        elif self.record_manager.status == RecordStatus.TELEOP:
            self.motion_manager.joint_pos = self.pred_action[:6]

        # Set gripper command
        if self.record_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.low[6]
        elif self.record_manager.status == RecordStatus.TELEOP:
            self.motion_manager.gripper_pos = self.pred_action[6]
