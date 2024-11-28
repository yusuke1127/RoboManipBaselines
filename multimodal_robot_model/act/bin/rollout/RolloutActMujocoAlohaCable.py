from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.rollout import RolloutMujocoAlohaCable


class RolloutActMujocoAlohaCable(RolloutAct, RolloutMujocoAlohaCable):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoAlohaCable()
    rollout.run()
