from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.rollout import RolloutMujocoXarm7Ring


class RolloutDiffusionPolicyMujocoXarm7Ring(
    RolloutDiffusionPolicy, RolloutMujocoXarm7Ring
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyMujocoXarm7Ring()
    rollout.run()
