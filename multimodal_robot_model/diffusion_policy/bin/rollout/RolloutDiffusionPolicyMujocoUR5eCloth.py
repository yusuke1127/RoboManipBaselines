from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eCloth

class RolloutDiffusionPolicyMujocoUR5eCloth(RolloutDiffusionPolicy, RolloutMujocoUR5eCloth):
    pass

if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eCloth()
    rollout.run()
