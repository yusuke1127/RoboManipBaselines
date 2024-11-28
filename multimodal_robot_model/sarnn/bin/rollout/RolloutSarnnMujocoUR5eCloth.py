from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eCloth


class RolloutSarnnMujocoUR5eCloth(RolloutSarnn, RolloutMujocoUR5eCloth):
    pass


if __name__ == "__main__":
    rollout = RolloutSarnnMujocoUR5eCloth()
    rollout.run()
