from isaacgym import (
    gymapi,  # noqa: F401
    gymtorch,  # noqa: F401
    gymutil,  # noqa: F401
)

from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutIsaacUR5eChain


class RolloutActIsaacUR5eChain(RolloutAct, RolloutIsaacUR5eChain):
    pass


if __name__ == "__main__":
    rollout = RolloutActIsaacUR5eChain()
    rollout.run()
