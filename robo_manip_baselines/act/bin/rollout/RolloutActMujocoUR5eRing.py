from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing


class RolloutActMujocoUR5eRing(RolloutAct, RolloutMujocoUR5eRing):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eRing()
    rollout.run()
