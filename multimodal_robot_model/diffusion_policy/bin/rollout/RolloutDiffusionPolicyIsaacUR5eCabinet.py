from isaacgym import gymapi  # noqa: F401
from isaacgym import gymutil  # noqa: F401
from isaacgym import gymtorch  # noqa: F401

from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.rollout import RolloutIsaacUR5eCabinet


class RolloutDiffusionPolicyIsaacUR5eCabinet(
    RolloutDiffusionPolicy, RolloutIsaacUR5eCabinet
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyIsaacUR5eCabinet()
    rollout.run()
