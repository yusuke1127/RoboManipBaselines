from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.tasks import RolloutUR5eCloth

class RolloutSarnnUR5eCloth(RolloutSarnn, RolloutUR5eCloth):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnUR5eCloth()
    rollout.run()
