from robo_manip_baselines.common.rollout import RolloutMujocoAlohaCable
from robo_manip_baselines.sarnn import RolloutSarnn


class RolloutSarnnMujocoAlohaCable(RolloutSarnn, RolloutMujocoAlohaCable):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoAlohaCable()
    rollout.run()
