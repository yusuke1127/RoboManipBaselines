from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.tasks import RolloutUR5eRing

class RolloutSarnnUR5eRing(RolloutSarnn, RolloutUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnUR5eRing()
    rollout.run()
