from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Cable
from robo_manip_baselines.sarnn import RolloutSarnn


class RolloutSarnnMujocoXarm7Cable(RolloutSarnn, RolloutMujocoXarm7Cable):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoXarm7Cable()
    rollout.run()
