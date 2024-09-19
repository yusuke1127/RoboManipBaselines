import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from TeleopBase import TeleopBase
from multimodal_robot_model.common import RecordStatus

class TeleopMujocoUR5eRing(TeleopBase):
    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/MujocoUR5eRingEnv-v0",
            render_mode="human",
            extra_camera_configs=[
                {"name": "front", "size": (480, 640)},
                {"name": "side", "size": (480, 640)},
                {"name": "hand", "size": (480, 640)},
            ]
        )
        self.demo_name = "MujocoUR5eRing"

    def setArmCommand(self):
        if self.record_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            target_pos = 0.5 * (self.env.unwrapped.get_geom_pose("fook1")[0:3] +
                                self.env.unwrapped.get_geom_pose("fook2")[0:3])
            if self.record_manager.status == RecordStatus.PRE_REACH:
                target_pos += np.array([-0.15, 0.05, -0.05]) # [m]
            elif self.record_manager.status == RecordStatus.REACH:
                target_pos += np.array([-0.1, 0.05, -0.05]) # [m]
            self.motion_manager.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(np.pi/2, 0.0, np.pi/2), target_pos)
        else:
            super().setArmCommand()

if __name__ == "__main__":
    teleop = TeleopMujocoUR5eRing()
    teleop.run()
