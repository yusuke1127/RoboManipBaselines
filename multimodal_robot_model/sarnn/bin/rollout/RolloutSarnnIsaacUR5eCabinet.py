from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutIsaacUR5eCabinet


class RolloutSarnnIsaacUR5eCabinet(RolloutSarnn, RolloutIsaacUR5eCabinet):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnIsaacUR5eCabinet()
    rollout.run()
