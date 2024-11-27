from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutMujocoXarm7Ring

class RolloutSarnnMujocoXarm7Ring(RolloutSarnn, RolloutMujocoXarm7Ring):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnMujocoXarm7Ring()
    rollout.run()
