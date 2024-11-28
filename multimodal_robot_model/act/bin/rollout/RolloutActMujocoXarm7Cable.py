from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.rollout import RolloutMujocoXarm7Cable

class RolloutActMujocoXarm7Cable(RolloutAct, RolloutMujocoXarm7Cable):
    pass

if __name__ == "__main__":
    rollout = RolloutActMujocoXarm7Cable()
    rollout.run()
