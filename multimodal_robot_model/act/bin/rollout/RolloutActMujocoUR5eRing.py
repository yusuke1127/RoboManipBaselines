from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutMujocoUR5eRing

class RolloutActMujocoUR5eRing(RolloutAct, RolloutMujocoUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eRing()
    rollout.run()
