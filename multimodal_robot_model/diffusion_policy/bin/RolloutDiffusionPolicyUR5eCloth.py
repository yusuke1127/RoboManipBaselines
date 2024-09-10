from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.tasks import RolloutUR5eCloth

class RolloutDiffusionPolicyUR5eCloth(RolloutDiffusionPolicy, RolloutUR5eCloth):
    pass

if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyUR5eCloth()
    rollout.run()
