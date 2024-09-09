from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutUR5eCable

class RolloutActUR5eCable(RolloutAct, RolloutUR5eCable):
    pass

if __name__ == "__main__":
    rollout = RolloutActUR5eCable()
    rollout.run()
