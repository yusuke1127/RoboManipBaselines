from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eRing


class RolloutDiffusionPolicyMujocoUR5eRing(
    RolloutDiffusionPolicy, RolloutMujocoUR5eRing
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eRing()
    rollout.run()
