from robo_manip_baselines.common.rollout import RolloutMujocoAlohaCable
from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy


class RolloutDiffusionPolicyMujocoAlohaCable(
    RolloutDiffusionPolicy, RolloutMujocoAlohaCable
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoAlohaCable()
    rollout.run()
