from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Ring
from robo_manip_baselines.sarnn import RolloutSarnn


class RolloutSarnnMujocoXarm7Ring(RolloutSarnn, RolloutMujocoXarm7Ring):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoXarm7Ring()
    rollout.run()
