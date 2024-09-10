from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.tasks import RolloutUR5eCloth

class RolloutMtActUR5eCloth(RolloutMtAct, RolloutUR5eCloth):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActUR5eCloth()
    rollout.run()
