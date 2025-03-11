from abc import ABCMeta, abstractmethod


class InputDeviceBase(metaclass=ABCMeta):
    """Base class for teleoperation input device."""

    def __init__(self, motion_manager):
        self.motion_manager = motion_manager

        self.connected = False
        self.state = None

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def set_arm_command(self):
        pass

    @abstractmethod
    def set_gripper_command(self):
        pass
