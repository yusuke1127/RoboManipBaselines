from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.tasks import RolloutUR5eRing

class RolloutDiffusionPolicyUR5eRing(RolloutDiffusionPolicy, RolloutUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyUR5eRing()
    rollout.run()
