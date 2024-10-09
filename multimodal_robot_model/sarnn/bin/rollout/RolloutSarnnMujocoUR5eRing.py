from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eRing

class RolloutSarnnMujocoUR5eRing(RolloutSarnn, RolloutMujocoUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eRing()
    rollout.run()
