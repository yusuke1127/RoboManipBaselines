import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from DemoTeleopBase import DemoTeleopBase
from DemoUtils import RecordStatus

class DemoTeleopUR5eRing(DemoTeleopBase):
    def __init__(self):
        env = gym.make(
            "multimodal_robot_model/UR5eRingEnv-v0",
            render_mode="human",
            extra_camera_configs=[
                {"name": "front", "size": (480, 640)},
                {"name": "side", "size": (480, 640)},
                {"name": "hand", "size": (480, 640)},
            ]
        )
        super().__init__(env, "UR5eRing")

    def setArmCommand(self):
        if self.record_manager.status == RecordStatus.PRE_REACH:
            target_pos = 0.5 * (self.env.unwrapped.data.geom("fook1").xpos +
                                self.env.unwrapped.data.geom("fook2").xpos)
            target_pos += np.array([-0.15, 0.05, -0.05]) # [m]
            self.motion_manager.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, np.pi/2), target_pos)
        elif self.record_manager.status == RecordStatus.REACH:
            target_pos = 0.5 * (self.env.unwrapped.data.geom("fook1").xpos +
                                self.env.unwrapped.data.geom("fook2").xpos)
            target_pos += np.array([-0.1, 0.05, -0.05]) # [m]
            self.motion_manager.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, np.pi/2), target_pos)
        else:
            super().setArmCommand()

if __name__ == "__main__":
    demo_teleop = DemoTeleopUR5eRing()
    demo_teleop.run()
