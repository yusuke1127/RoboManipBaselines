from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.tasks import RolloutMujocoUR5eCable

class RolloutMtActMujocoUR5eCable(RolloutMtAct, RolloutMujocoUR5eCable):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eCable()
    rollout.run()
