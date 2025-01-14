from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing
from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy


class RolloutDiffusionPolicyMujocoUR5eRing(
    RolloutDiffusionPolicy, RolloutMujocoUR5eRing
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eRing()
    rollout.run()
