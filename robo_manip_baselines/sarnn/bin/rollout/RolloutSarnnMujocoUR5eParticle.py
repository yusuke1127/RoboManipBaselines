from robo_manip_baselines.sarnn import RolloutSarnn
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eParticle


class RolloutSarnnMujocoUR5eParticle(RolloutSarnn, RolloutMujocoUR5eParticle):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eParticle()
    rollout.run()
