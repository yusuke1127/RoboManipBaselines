from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCloth


class RolloutActMujocoUR5eCloth(RolloutAct, RolloutMujocoUR5eCloth):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eCloth()
    rollout.run()
