from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.rollout import RolloutMujocoAlohaCable


class RolloutDiffusionPolicyMujocoAlohaCable(
    RolloutDiffusionPolicy, RolloutMujocoAlohaCable
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoAlohaCable()
    rollout.run()
