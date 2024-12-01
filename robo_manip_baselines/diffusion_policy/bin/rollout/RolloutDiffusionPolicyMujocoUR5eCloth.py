from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCloth


class RolloutDiffusionPolicyMujocoUR5eCloth(
    RolloutDiffusionPolicy, RolloutMujocoUR5eCloth
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eCloth()
    rollout.run()
