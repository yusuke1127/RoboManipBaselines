from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.tasks import RolloutUR5eParticle

class RolloutActUR5eParticle(RolloutAct, RolloutUR5eParticle):
    pass

if __name__ == "__main__":
    rollout = RolloutActUR5eParticle()
    rollout.run()
