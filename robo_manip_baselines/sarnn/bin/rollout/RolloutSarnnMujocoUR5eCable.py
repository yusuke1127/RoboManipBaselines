from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCable


class RolloutSarnnMujocoUR5eCable(RolloutSarnn, RolloutMujocoUR5eCable):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eCable()
    rollout.run()
