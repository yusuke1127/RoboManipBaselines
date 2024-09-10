from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.tasks import RolloutUR5eRing

class RolloutMtActUR5eRing(RolloutMtAct, RolloutUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActUR5eRing()
    rollout.run()
