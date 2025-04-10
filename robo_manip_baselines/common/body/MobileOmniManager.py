import dataclasses

import numpy as np

from ..data.DataKey import DataKey
from .BodyManagerBase import BodyConfigBase, BodyManagerBase


class MobileOmniManager(BodyManagerBase):
    """Manager for omni-directional mobile base."""

    SUPPORTED_DATA_KEYS = [
        DataKey.MEASURED_MOBILE_OMNI_VEL,
        DataKey.COMMAND_MOBILE_OMNI_VEL,
    ]

    def reset(self, init=False):
        self.target_vel = np.zeros(3)

    def set_command_data(self, key, command, is_skip=False):
        if key == DataKey.COMMAND_MOBILE_OMNI_VEL:
            self.target_vel = command
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def get_command_data(self, key):
        if key == DataKey.COMMAND_MOBILE_OMNI_VEL:
            return self.target_vel
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def draw_markers(self):
        pass


@dataclasses.dataclass
class MobileOmniConfig(BodyConfigBase):
    """Configuration for omni-directional mobile base."""

    BodyManagerClass = MobileOmniManager
