from abc import ABCMeta, abstractmethod


class InputDeviceBase(metaclass=ABCMeta):
    """Base class for teleoperation input device."""

    def __init__(self):
        self.connected = False
        self.state = None

    @abstractmethod
    def connect(self, motion_manager):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def set_arm_command(self, motion_manager):
        pass

    @abstractmethod
    def set_gripper_command(self, motion_manager):
        pass
