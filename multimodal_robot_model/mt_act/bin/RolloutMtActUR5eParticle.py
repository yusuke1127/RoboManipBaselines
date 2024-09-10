from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.tasks import RolloutUR5eParticle

class RolloutMtActUR5eParticle(RolloutMtAct, RolloutUR5eParticle):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActUR5eParticle()
    rollout.run()
