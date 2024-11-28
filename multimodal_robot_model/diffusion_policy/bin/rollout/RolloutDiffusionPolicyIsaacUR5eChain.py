from isaacgym import gymapi  # noqa: F401
from isaacgym import gymutil  # noqa: F401
from isaacgym import gymtorch  # noqa: F401

from multimodal_robot_model.diffusion_policy import RolloutDiffusionPolicy
from multimodal_robot_model.common.rollout import RolloutIsaacUR5eChain


class RolloutDiffusionPolicyIsaacUR5eChain(
    RolloutDiffusionPolicy, RolloutIsaacUR5eChain
):
    pass


if __name__ == "__main__":
    rollout = RolloutDiffusionPolicyIsaacUR5eChain()
    rollout.run()
