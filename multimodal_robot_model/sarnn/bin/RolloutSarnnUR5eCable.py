from multimodal_robot_model.sarnn import RolloutSarnn
from multimodal_robot_model.common.tasks import RolloutUR5eCable

class RolloutSarnnUR5eCable(RolloutSarnn, RolloutUR5eCable):
    pass

if __name__ == "__main__":
    rollout = RolloutSarnnUR5eCable()
    rollout.run()
