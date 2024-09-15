from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.tasks import RolloutMujocoUR5eRing

class RolloutDiffusionPolicyMujocoUR5eRing(RolloutDiffusionPolicy, RolloutMujocoUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eRing()
    rollout.run()
