from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutMujocoXarm7Cable

class RolloutSarnnMujocoXarm7Cable(RolloutSarnn, RolloutMujocoXarm7Cable):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnMujocoXarm7Cable()
    rollout.run()
