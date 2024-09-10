from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutUR5eCloth

class RolloutActUR5eCloth(RolloutAct, RolloutUR5eCloth):
    pass

if __name__ == "__main__":
    rollout = RolloutActUR5eCloth()
    rollout.run()
