import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from DemoTeleopBase import DemoTeleopBase
from DemoUtils import RecordStatus

class DemoTeleopUR5eCable(DemoTeleopBase):
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
        super().__init__(env, "UR5eCable")

    def setArmCommand(self):
        if self.record_manager.status == RecordStatus.PRE_REACH:
            target_pos = self.env.unwrapped.model.body("cable_end").pos.copy()
            target_pos[2] = 1.02 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
        elif self.record_manager.status == RecordStatus.REACH:
            target_pos = self.env.unwrapped.model.body("cable_end").pos.copy()
            target_pos[2] = 0.995 # [m]
            self.motion_manager.target_se3 = pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)
        else:
            super().setArmCommand()

if __name__ == "__main__":
    demo_teleop = DemoTeleopUR5eCable()
    demo_teleop.run()
