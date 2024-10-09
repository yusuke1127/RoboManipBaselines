from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eCable

class RolloutSarnnMujocoUR5eCable(RolloutSarnn, RolloutMujocoUR5eCable):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eCable()
    rollout.run()
