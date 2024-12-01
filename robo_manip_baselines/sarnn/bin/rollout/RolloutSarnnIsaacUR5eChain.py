from isaacgym import gymapi  # noqa: F401
from isaacgym import gymutil  # noqa: F401
from isaacgym import gymtorch  # noqa: F401

from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutIsaacUR5eChain


class RolloutSarnnIsaacUR5eChain(RolloutSarnn, RolloutIsaacUR5eChain):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnIsaacUR5eChain()
    rollout.run()
