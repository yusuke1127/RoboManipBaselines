from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Ring
from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy


class RolloutDiffusionPolicyMujocoXarm7Ring(
    RolloutDiffusionPolicy, RolloutMujocoXarm7Ring
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoXarm7Ring()
    rollout.run()
