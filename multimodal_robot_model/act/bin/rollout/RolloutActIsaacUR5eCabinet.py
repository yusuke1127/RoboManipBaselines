from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.rollout import RolloutIsaacUR5eCabinet

class RolloutActIsaacUR5eCabinet(RolloutAct, RolloutIsaacUR5eCabinet):
    pass

if __name__ == "__main__":
    rollout = RolloutActIsaacUR5eCabinet()
    rollout.run()
