from robo_manip_baselines.common.rollout import RolloutMujocoUR5eParticle
from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy


class RolloutDiffusionPolicyMujocoUR5eParticle(
    RolloutDiffusionPolicy, RolloutMujocoUR5eParticle
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eParticle()
    rollout.run()
