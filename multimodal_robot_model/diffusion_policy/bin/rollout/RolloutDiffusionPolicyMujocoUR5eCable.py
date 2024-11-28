from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eCable


class RolloutDiffusionPolicyMujocoUR5eCable(
    RolloutDiffusionPolicy, RolloutMujocoUR5eCable
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoUR5eCable()
    rollout.run()
