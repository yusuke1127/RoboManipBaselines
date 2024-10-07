import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from multimodal_robot_model.teleop import TeleopBase
from multimodal_robot_model.common import MotionStatus

class TeleopRealUR5eGear(TeleopBase):
    def __init__(self, robot_ip, camera_ids):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        super().__init__()

    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/RealUR5eGearEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids
        )
        self.demo_name = "RealUR5eGear"

    def setArmCommand(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            # No action is required in pre-reach or reach phases
            pass
        else:
            super().setArmCommand()

    def setGripperCommand(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.low[6]
        else:
            super().setGripperCommand()

if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    camera_ids = {"front": "832112072660",
                  "side": None,
                  "hand": None}
    teleop = TeleopRealUR5eGear(robot_ip, camera_ids)
    teleop.run()
