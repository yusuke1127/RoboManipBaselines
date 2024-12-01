from isaacgym import gymapi  # noqa: F401
from isaacgym import gymutil  # noqa: F401
from isaacgym import gymtorch  # noqa: F401

from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutIsaacUR5eCabinet


class RolloutSarnnIsaacUR5eCabinet(RolloutSarnn, RolloutIsaacUR5eCabinet):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnIsaacUR5eCabinet()
    rollout.run()
