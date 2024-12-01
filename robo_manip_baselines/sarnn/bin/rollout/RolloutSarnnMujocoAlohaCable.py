from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoAlohaCable


class RolloutSarnnMujocoAlohaCable(RolloutSarnn, RolloutMujocoAlohaCable):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoAlohaCable()
    rollout.run()
