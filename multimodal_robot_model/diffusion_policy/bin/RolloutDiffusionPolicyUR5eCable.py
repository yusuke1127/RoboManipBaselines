from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.tasks import RolloutUR5eCable

class RolloutDiffusionPolicyUR5eCable(RolloutDiffusionPolicy, RolloutUR5eCable):
    pass

if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyUR5eCable()
    rollout.run()
