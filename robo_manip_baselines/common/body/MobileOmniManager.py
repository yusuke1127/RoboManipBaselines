import dataclasses

from .BodyManagerBase import BodyConfigBase, BodyManagerBase


class MobileOmniManager(BodyManagerBase):
    """Manager for omni-directional mobile base."""

    SUPPORTED_DATA_KEYS = []


@dataclasses.dataclass
class MobileOmniConfig(BodyConfigBase):
    """Configuration for omni-directional mobile base."""

    BodyManagerClass = MobileOmniManager
