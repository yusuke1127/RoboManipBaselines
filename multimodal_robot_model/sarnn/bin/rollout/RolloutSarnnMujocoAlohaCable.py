from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutMujocoAlohaCable

class RolloutSarnnMujocoAlohaCable(RolloutSarnn, RolloutMujocoAlohaCable):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnMujocoAlohaCable()
    rollout.run()
