from os import path
import time
import numpy as np
from gymnasium.spaces import Box, Dict

from ..RealEnvBase import RealEnvBase

import rtde_control
import rtde_receive
from gello.robots.robotiq_gripper import RobotiqGripper


class RealUR5eEnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [
                -2 * np.pi,
                -2 * np.pi,
                -1 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                2 * np.pi,
                2 * np.pi,
                1 * np.pi,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                255.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip,
        camera_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.gripper_action_idxes = [6]
        self.arm_action_idxes = slice(0, 6)
        self.arm_urdf_path = path.join(
            path.dirname(__file__), "../../assets/common/robots/ur5e/ur5e.urdf"
        )
        self.arm_root_pose = None
        self.ik_eef_joint_id = 6
        self.init_qpos = init_qpos
        self.qvel_limit = np.deg2rad(191)  # [rad/s]

        # Connect to UR5e
        print("[RealUR5eEnvBase] Start connecting the UR5e.")
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c.endFreedriveMode()
        self.arm_qpos_actual = np.array(self.rtde_r.getActualQ())
        print("[RealUR5eEnvBase] Finish connecting the UR5e.")

        # Connect to Robotiq gripper
        print("[RealUR5eEnvBase] Start connecting the Robotiq gripper.")
        self.gripper_port = 63352
        self.gripper = RobotiqGripper()
        self.gripper.connect(hostname=self.robot_ip, port=self.gripper_port)
        self._gripper_activated = False
        print("[RealUR5eEnvBase] Finish connecting the Robotiq gripper.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)

    def _reset_robot(self):
        print("[RealUR5eEnvBase] Start moving the robot to the reset position.")
        self._set_action(self.init_qpos, duration=None, qvel_limit_scale=0.3, wait=True)
        print("[RealUR5eEnvBase] Finish moving the robot to the reset position.")

        if not self._gripper_activated:
            self._gripper_activated = True
            print("[RealUR5eEnvBase] Start activating the Robotiq gripper.")
            self.gripper.activate()
            print("[RealUR5eEnvBase] Finish activating the Robotiq gripper.")

    def _set_action(self, action, duration=None, qvel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or qpos for safety
        arm_qpos_command = action[self.arm_action_idxes]
        scaled_qvel_limit = np.clip(qvel_limit_scale, 0.01, 10.0) * self.qvel_limit
        if duration is None:
            duration_min, duration_max = 0.1, 10.0  # [s]
            duration = np.clip(
                np.max(
                    np.abs(arm_qpos_command - self.arm_qpos_actual) / scaled_qvel_limit
                ),
                duration_min,
                duration_max,
            )
        else:
            arm_qpos_command_overwritten = self.arm_qpos_actual + np.clip(
                arm_qpos_command - self.arm_qpos_actual,
                -1 * scaled_qvel_limit * duration,
                scaled_qvel_limit * duration,
            )
            # if np.linalg.norm(arm_qpos_command_overwritten - arm_qpos_command) > 1e-10:
            #     print("[RealUR5eEnvBase] Overwrite joint command for safety.")
            arm_qpos_command = arm_qpos_command_overwritten

        # Send command to UR5e
        velocity = 0.5
        acceleration = 0.5
        lookahead_time = 0.2  # [s]
        gain = 100
        period = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(
            arm_qpos_command, velocity, acceleration, duration, lookahead_time, gain
        )
        self.rtde_c.waitPeriod(period)

        # Send command to Robotiq gripper
        gripper_pos = action[self.gripper_action_idxes]
        speed = 50
        force = 10
        self.gripper.move(int(gripper_pos[0]), speed, force)

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from UR5e
        arm_qpos = np.array(self.rtde_r.getActualQ())
        arm_qvel = np.array(self.rtde_r.getActualQd())
        self.arm_qpos_actual = arm_qpos.copy()

        # Get state from Robotiq gripper
        gripper_pos = np.array([self.gripper.get_current_position()], dtype=np.float64)
        gripper_vel = np.zeros(1)

        # Get wrench from force sensor
        # Set zero because UR5e does not have a wrist force sensor
        force = np.zeros(3)
        torque = np.zeros(3)

        return {
            "joint_pos": np.concatenate((arm_qpos, gripper_pos), dtype=np.float64),
            "joint_vel": np.concatenate((arm_qvel, gripper_vel), dtype=np.float64),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
        }
