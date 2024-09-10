import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.demos.DemoUtils import MotionManager, RecordStatus, RecordManager
from ..RolloutBase import RolloutBase

class RolloutUR5eRing(RolloutBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/UR5eRingEnv-v0",
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
            target_pos = 0.5 * (self.env.unwrapped.data.geom("fook1").xpos +
                                self.env.unwrapped.data.geom("fook2").xpos)
            if self.record_manager.status == RecordStatus.PRE_REACH:
                target_pos += np.array([-0.15, 0.05, -0.05]) # [m]
            elif self.record_manager.status == RecordStatus.REACH:
                target_pos += np.array([-0.1, 0.05, -0.05]) # [m]
            self.motion_manager.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, np.pi/2), target_pos)
            self.motion_manager.inverseKinematics()
        elif self.record_manager.status == RecordStatus.TELEOP:
            self.motion_manager.joint_pos = self.pred_action[:6]

        # Set gripper command
        if self.record_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.high[6]
        elif self.record_manager.status == RecordStatus.TELEOP:
            self.motion_manager.gripper_pos = self.pred_action[6]
