import glob

import numpy as np

from robo_manip_baselines.common import DataKey

from .InputDeviceBase import InputDeviceBase


class GelloInputDevice(InputDeviceBase):
    """
    GELLO for teleoperation input device.

    Ref: https://github.com/wuphilipp/gello_software/blob/daae81f4a78ec0f7534937413345d96d3e1bc7fc/experiments/run_env.py
    """

    PORT_CONFIG_MAP = {
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9MIQNO-if00-port0": {
            "joint_ids": (1, 2, 3, 4, 5, 6),
            "joint_offsets": (
                2 * np.pi / 2,
                3 * np.pi / 2,
                2 * np.pi / 2,
                -1 * np.pi / 2,
                2 * np.pi / 2,
                4 * np.pi / 2,
            ),
            "joint_signs": (1, 1, -1, 1, 1, 1),
            "gripper_config": (7, 197.378125, 155.578125),
        }
    }

    def __init__(self, motion_manager, port=None):
        super().__init__(motion_manager)

        self.port = port
        self.interp_end_time_idx = 50

    def connect(self):
        if self.connected:
            return

        self.connected = True

        from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent

        # Set port
        if self.port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) > 0:
                self.port = usb_ports[0]
                print(
                    f"[{self.__class__.__name__}] Found {len(usb_ports)} ports in /dev/serial/by-id/*."
                )
            else:
                raise RuntimeError(f"[{self.__class__.__name__}] No port found.")
        print(f"[{self.__class__.__name__}] Connect to {self.port}")

        # Instantiate gello agent
        dynamixel_config = DynamixelRobotConfig(**self.PORT_CONFIG_MAP[self.port])
        current_joint_pos = self.motion_manager.get_command_data(
            DataKey.COMMAND_JOINT_POS
        )
        self.agent = GelloAgent(
            port=self.port,
            dynamixel_config=dynamixel_config,
            start_joints=current_joint_pos,
        )
        self.time_idx = 0

        # Check joint error
        new_joint_pos = self.get_joint_pos()
        if current_joint_pos.shape != new_joint_pos.shape:
            raise RuntimeError(
                f"[{self.__class__.__name__}] The shape of current_joint_pos and new_joint_pos do not match: {current_joint_pos.shape} != {new_joint_pos.shape}"
            )
        joint_pos_thre = np.deg2rad(60.0)
        if np.max(np.abs(current_joint_pos - new_joint_pos)) > joint_pos_thre:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Joint angles differ greatly:\n  robot: {current_joint_pos}\n  gello: {new_joint_pos}"
            )

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        self.state = self.get_joint_pos()

    def set_arm_command(self):
        new_joint_pos = self.state.copy()
        current_joint_pos = self.motion_manager.get_command_data(
            DataKey.COMMAND_JOINT_POS
        )

        # Keep the current gripper command
        gripper_joint_idxes = self.motion_manager.env.unwrapped.gripper_joint_idxes
        new_joint_pos[gripper_joint_idxes] = current_joint_pos[gripper_joint_idxes]

        # Interpolate command
        if self.time_idx < self.interp_end_time_idx:
            interp_ratio = self.time_idx / self.interp_end_time_idx
            new_joint_pos = (
                interp_ratio * new_joint_pos + (1 - interp_ratio) * current_joint_pos
            )

        self.motion_manager.set_command_data(DataKey.COMMAND_JOINT_POS, new_joint_pos)
        self.time_idx += 1

    def set_gripper_command(self):
        gripper_joint_pos = self.state[
            self.motion_manager.env.unwrapped.gripper_joint_idxes
        ]
        self.motion_manager.set_command_data(
            DataKey.COMMAND_GRIPPER_JOINT_POS, gripper_joint_pos
        )

    def get_joint_pos(self):
        gripper_joint_idxes = self.motion_manager.env.unwrapped.gripper_joint_idxes
        gripper_joint_pos_low = self.motion_manager.env.action_space.low[
            gripper_joint_idxes
        ]
        gripper_joint_pos_high = self.motion_manager.env.action_space.high[
            gripper_joint_idxes
        ]

        joint_pos = self.agent.act().copy()
        joint_pos[gripper_joint_idxes] = (
            gripper_joint_pos_high - gripper_joint_pos_low
        ) * joint_pos[gripper_joint_idxes] + gripper_joint_pos_low

        return joint_pos
