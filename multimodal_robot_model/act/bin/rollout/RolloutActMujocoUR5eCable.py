from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutMujocoUR5eCable

class RolloutActMujocoUR5eCable(RolloutAct, RolloutMujocoUR5eCable):
    pass

if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eCable()
    rollout.run()
