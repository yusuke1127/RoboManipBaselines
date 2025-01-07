from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eInsert


class RolloutSarnnMujocoUR5eInsert(RolloutSarnn, RolloutMujocoUR5eInsert):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eInsert()
    rollout.run()
