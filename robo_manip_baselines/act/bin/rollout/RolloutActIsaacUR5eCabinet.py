from isaacgym import gymapi  # noqa: F401
from isaacgym import gymutil  # noqa: F401
from isaacgym import gymtorch  # noqa: F401

from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutIsaacUR5eCabinet


class RolloutActIsaacUR5eCabinet(RolloutAct, RolloutIsaacUR5eCabinet):
    pass


if __name__ == "__main__":
    rollout = RolloutActIsaacUR5eCabinet()
    rollout.run()
