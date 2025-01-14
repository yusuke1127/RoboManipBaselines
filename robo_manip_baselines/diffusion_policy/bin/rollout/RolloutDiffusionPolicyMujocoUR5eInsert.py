from robo_manip_baselines.common.rollout import RolloutMujocoUR5eInsert
from robo_manip_baselines.diffusion_policy import RolloutDiffusionPolicy


class RolloutDiffusionPolicyMujocoUR5eInsert(
    RolloutDiffusionPolicy, RolloutMujocoUR5eInsert
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eInsert()
    rollout.run()
