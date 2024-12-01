from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoAlohaCable


class RolloutActMujocoAlohaCable(RolloutAct, RolloutMujocoAlohaCable):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoAlohaCable()
    rollout.run()
