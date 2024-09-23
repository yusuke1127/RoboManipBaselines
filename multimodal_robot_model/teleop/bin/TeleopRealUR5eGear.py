import numpy as np
import gymnasium as gym
import pinocchio as pin
import multimodal_robot_model
from multimodal_robot_model.teleop import TeleopBase
from multimodal_robot_model.common import RecordStatus

class TeleopRealUR5eGear(TeleopBase):
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        super().__init__()

    def setupEnv(self):
        self.env = gym.make(
            "multimodal_robot_model/RealUR5eGearEnv-v0",
            robot_ip=self.robot_ip
        )
        self.demo_name = "RealUR5eGear"

    def setArmCommand(self):
        if self.record_manager.status in (RecordStatus.PRE_REACH, RecordStatus.REACH):
            # No action is required in pre-reach or reach phases
            pass
        else:
            super().setArmCommand()

    def setGripperCommand(self):
        if self.record_manager.status == RecordStatus.GRASP:
            self.motion_manager.gripper_pos = self.env.action_space.low[6]
        else:
            super().setGripperCommand()

if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    teleop = TeleopRealUR5eGear(robot_ip)
    teleop.run()
