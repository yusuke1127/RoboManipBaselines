from robo_manip_baselines.mt_act import RolloutMtAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCloth


class RolloutMtActMujocoUR5eCloth(RolloutMtAct, RolloutMujocoUR5eCloth):
    pass


if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eCloth()
    rollout.run()
