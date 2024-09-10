import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from DemoTeleopBase import DemoTeleopBase
from DemoUtils import RecordStatus

class DemoTeleopUR5eParticle(DemoTeleopBase):
    def __init__(self):
        super().__init__()

        # Command configuration
        self.command_rpy_scale = 1e-2

    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/UR5eParticleEnv-v0",
            render_mode="human",
            extra_camera_configs=[
                {"name": "front", "size": (480, 640)},
                {"name": "side", "size": (480, 640)},
                {"name": "hand", "size": (480, 640)},
            ]
        )
        self.demo_name = "UR5eParticle"

    def setArmCommand(self):
        if self.record_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            target_pos = self.env.unwrapped.data.geom("scoop_handle").xpos
            if self.record_manager.status == RecordStatus.PRE_REACH:
                target_pos += np.array([0.0, 0.0, 0.2]) # [m]
            elif self.record_manager.status == RecordStatus.REACH:
                target_pos += np.array([0.0, 0.0, 0.15]) # [m]
            self.motion_manager.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi, 0.0, np.pi/2), target_pos)
        else:
            super().setArmCommand()

if __name__ == "__main__":
    demo_teleop = DemoTeleopUR5eParticle()
    demo_teleop.run()
