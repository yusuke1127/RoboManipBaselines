import numpy as np
import gymnasium as gym
from robo_manip_baselines.teleop import TeleopBase
from robo_manip_baselines.common import MotionStatus


class TeleopRealUR5eDemo(TeleopBase):
    def __init__(self, robot_ip, camera_ids):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        super().__init__()

    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/RealUR5eDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
        )
        self.demo_name = self.args.demo_name or "RealUR5eDemo"

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            # No action is required in pre-reach or reach phases
            pass
        else:
            super().set_arm_command()

    def set_gripper_command(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_joint_pos = np.array([150.0])
        else:
            super().set_gripper_command()


if __name__ == "__main__":
    robot_ip = "192.168.11.4"
    camera_ids = {"front": "145522067924", "side": None, "hand": "153122070885"}
    teleop = TeleopRealUR5eDemo(robot_ip, camera_ids)
    teleop.run()
