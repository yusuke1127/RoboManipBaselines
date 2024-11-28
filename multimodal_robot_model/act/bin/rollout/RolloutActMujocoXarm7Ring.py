from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.rollout import RolloutMujocoXarm7Ring

class RolloutActMujocoXarm7Ring(RolloutAct, RolloutMujocoXarm7Ring):
    pass

if __name__ == "__main__":
    rollout = RolloutActMujocoXarm7Ring()
    rollout.run()
