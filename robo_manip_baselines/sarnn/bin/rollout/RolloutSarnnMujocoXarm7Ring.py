from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Ring


class RolloutSarnnMujocoXarm7Ring(RolloutSarnn, RolloutMujocoXarm7Ring):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoXarm7Ring()
    rollout.run()
