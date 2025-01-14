from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Cable
from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy


class RolloutDiffusionPolicyMujocoXarm7Cable(
    RolloutDiffusionPolicy, RolloutMujocoXarm7Cable
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoXarm7Cable()
    rollout.run()
