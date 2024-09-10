from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.tasks import RolloutUR5eCable

class RolloutMtActUR5eCable(RolloutMtAct, RolloutUR5eCable):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActUR5eCable()
    rollout.run()
