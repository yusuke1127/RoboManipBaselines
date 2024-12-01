from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy
from robo_manip_baselines.common.rollout import RolloutMujocoAlohaCable


class RolloutDiffusionPolicyMujocoAlohaCable(
    RolloutDiffusionPolicy, RolloutMujocoAlohaCable
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoAlohaCable()
    rollout.run()
