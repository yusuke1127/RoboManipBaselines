import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from DemoTeleopBase import DemoTeleopBase
from DemoUtils import RecordStatus

class DemoTeleopUR5eCloth(DemoTeleopBase):
    def __init__(self):
        env = gym.make(
            "multimodal_robot_model/UR5eClothEnv-v0",
            render_mode="human",
            extra_camera_configs=[
                {"name": "front", "size": (480, 640)},
                {"name": "side", "size": (480, 640)},
                {"name": "hand", "size": (480, 640)},
            ]
        )
        super().__init__(env, "UR5eCloth")

    def setArmCommand(self):
        if self.record_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, 0.25*np.pi),
                                 self.env.unwrapped.model.body("cloth_root_frame").pos.copy())
            if self.record_manager.status == RecordStatus.PRE_REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.25]))
            elif self.record_manager.status == RecordStatus.REACH:
                target_se3 *= pin.SE3(np.identity(3), np.array([0.0, -0.2, -0.2]))
            self.motion_manager.target_se3 = target_se3
        else:
            super().setArmCommand()

    def setGripperCommand(self):
        if self.record_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.low[6]
        else:
            super().setGripperCommand()

if __name__ == "__main__":
    demo_teleop = DemoTeleopUR5eCloth()
    demo_teleop.run()
