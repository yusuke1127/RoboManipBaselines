from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.tasks import RolloutUR5eParticle

class RolloutDiffusionPolicyUR5eParticle(RolloutDiffusionPolicy, RolloutUR5eParticle):
    pass

if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyUR5eParticle()
    rollout.run()
