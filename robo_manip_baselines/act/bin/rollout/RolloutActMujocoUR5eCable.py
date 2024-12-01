from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCable


class RolloutActMujocoUR5eCable(RolloutAct, RolloutMujocoUR5eCable):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eCable()
    rollout.run()
