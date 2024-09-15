from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutMujocoUR5eCloth

class RolloutActMujocoUR5eCloth(RolloutAct, RolloutMujocoUR5eCloth):
    pass

if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eCloth()
    rollout.run()
