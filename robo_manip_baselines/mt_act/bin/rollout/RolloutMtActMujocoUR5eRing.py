from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing
from robo_manip_baselines.mt_act import RolloutMtAct


class RolloutMtActMujocoUR5eRing(RolloutMtAct, RolloutMujocoUR5eRing):
    pass


if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eRing()
    rollout.run()
