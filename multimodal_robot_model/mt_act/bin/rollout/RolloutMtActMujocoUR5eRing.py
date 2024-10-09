from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eRing

class RolloutMtActMujocoUR5eRing(RolloutMtAct, RolloutMujocoUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eRing()
    rollout.run()
