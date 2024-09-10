from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutUR5eRing

class RolloutActUR5eRing(RolloutAct, RolloutUR5eRing):
    pass

if __name__ == "__main__":
    rollout = RolloutActUR5eRing()
    rollout.run()
