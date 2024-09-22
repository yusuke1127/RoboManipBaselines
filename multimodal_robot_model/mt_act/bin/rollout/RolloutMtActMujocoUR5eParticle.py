from multimodal_robot_model.mt_act import RolloutMtAct
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eParticle

class RolloutMtActMujocoUR5eParticle(RolloutMtAct, RolloutMujocoUR5eParticle):
    pass

if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eParticle()
    rollout.run()
