from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eCloth

class RolloutMtActMujocoUR5eCloth(RolloutMtAct, RolloutMujocoUR5eCloth):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eCloth()
    rollout.run()
