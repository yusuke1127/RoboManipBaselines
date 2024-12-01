from robo_manip_baselines.mt_act import RolloutMtAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing


class RolloutMtActMujocoUR5eRing(RolloutMtAct, RolloutMujocoUR5eRing):
    pass


if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eRing()
    rollout.run()
