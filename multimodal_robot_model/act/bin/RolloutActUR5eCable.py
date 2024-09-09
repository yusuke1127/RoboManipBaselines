import numpy as np
import pinocchio as pin
import gymnasium as gym
import multimodal_robot_model
from multimodal_robot_model.demos.DemoUtils import MotionManager, RecordStatus, RecordManager
from RolloutAct import RolloutAct

class RolloutActUR5eCable(RolloutAct):
    def __init__(self):
        env = gym.make(
            "multimodal_robot_model/UR5eCableEnv-v0",
            render_mode="human",
            extra_camera_configs=[
                {"name": "front", "size": (480, 640)},
                {"name": "side", "size": (480, 640)},
                {"name": "hand", "size": (480, 640)},
            ]
        )
        super().__init__(env, "ActUR5eCable")

    def setCommand(self):
        # Set joint command
        if self.record_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            target_pos = self.env.unwrapped.model.body("cable_end").pos.copy()
            if self.record_manager.status == RecordStatus.PRE_REACH:
                target_pos[2] = 1.02 # [m]
            elif self.record_manager.status == RecordStatus.REACH:
                target_pos[2] = 0.995 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
            self.motion_manager.inverseKinematics()
        elif self.record_manager.status == RecordStatus.TELEOP:
            self.motion_manager.joint_pos = self.pred_action[:6]

        # Set gripper command
        if self.record_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.high[6]
        elif self.record_manager.status == RecordStatus.TELEOP:
            self.motion_manager.gripper_pos = self.pred_action[6]

if __name__ == "__main__":
    rollout = RolloutActUR5eCable()
    rollout.run()
