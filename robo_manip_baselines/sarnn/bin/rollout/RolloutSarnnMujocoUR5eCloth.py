from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCloth


class RolloutSarnnMujocoUR5eCloth(RolloutSarnn, RolloutMujocoUR5eCloth):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eCloth()
    rollout.run()
