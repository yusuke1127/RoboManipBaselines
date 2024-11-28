from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eParticle


class RolloutDiffusionPolicyMujocoUR5eParticle(
    RolloutDiffusionPolicy, RolloutMujocoUR5eParticle
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eParticle()
    rollout.run()
