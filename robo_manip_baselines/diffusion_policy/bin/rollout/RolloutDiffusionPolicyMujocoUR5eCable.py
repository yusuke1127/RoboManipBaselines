from robo_manip_baselines.common.rollout import RolloutMujocoUR5eCable
from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy


class RolloutDiffusionPolicyMujocoUR5eCable(
    RolloutDiffusionPolicy, RolloutMujocoUR5eCable
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eCable()
    rollout.run()
