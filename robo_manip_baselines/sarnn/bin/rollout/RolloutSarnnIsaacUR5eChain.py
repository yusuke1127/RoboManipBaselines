from isaacgym import (
    gymapi,  # noqa: F401
    gymtorch,  # noqa: F401
    gymutil,  # noqa: F401
)

from robo_manip_baselines.common.rollout import RolloutIsaacUR5eChain
from robo_manip_baselines.sarnn import RolloutSarnn


class RolloutSarnnIsaacUR5eChain(RolloutSarnn, RolloutIsaacUR5eChain):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnIsaacUR5eChain()
    rollout.run()
