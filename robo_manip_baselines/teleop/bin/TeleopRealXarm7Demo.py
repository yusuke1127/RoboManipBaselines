import numpy as np
import gymnasium as gym
from robo_manip_baselines.teleop import TeleopBase
from robo_manip_baselines.common import MotionStatus


class TeleopRealXarm7Demo(TeleopBase):
    def __init__(self, robot_ip, camera_ids):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        super().__init__()

        # Command configuration
        self.gripper_scale = 10.0

    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/RealXarm7DemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
        )
        self.demo_name = self.args.demo_name or "RealXarm7Demo"

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            # No action is required in pre-reach or reach phases
            pass
        else:
            super().set_arm_command()

    def set_gripper_command(self):
        if self.data_manager.status == MotionStatus.GRASP:
            self.motion_manager.gripper_pos = np.array([800.0])
        else:
            super().set_gripper_command()


if __name__ == "__main__":
    robot_ip = "192.168.1.244"
    camera_ids = {"front": "314422070401", "side": None, "hand": "332522077926"}
    teleop = TeleopRealXarm7Demo(robot_ip, camera_ids)
    teleop.run()
