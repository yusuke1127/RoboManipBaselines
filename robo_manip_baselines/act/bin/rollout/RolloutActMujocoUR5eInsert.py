from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eInsert


class RolloutActMujocoUR5eInsert(RolloutAct, RolloutMujocoUR5eInsert):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eInsert()
    rollout.run()
