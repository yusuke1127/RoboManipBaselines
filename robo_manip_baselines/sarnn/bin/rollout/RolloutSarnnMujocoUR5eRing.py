from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing


class RolloutSarnnMujocoUR5eRing(RolloutSarnn, RolloutMujocoUR5eRing):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eRing()
    rollout.run()
