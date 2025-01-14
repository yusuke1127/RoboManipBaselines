from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing
from robo_manip_baselines.sarnn import RolloutSarnn


class RolloutSarnnMujocoUR5eRing(RolloutSarnn, RolloutMujocoUR5eRing):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eRing()
    rollout.run()
