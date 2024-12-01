from robo_manip_baselines.mt_act import RolloutMtAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCable


class RolloutMtActMujocoUR5eCable(RolloutMtAct, RolloutMujocoUR5eCable):
    pass


if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eCable()
    rollout.run()
