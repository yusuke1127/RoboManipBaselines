from multimodal_robot_model.act import RolloutAct
from multimodal_robot_model.common.rollout import RolloutMujocoUR5eParticle


class RolloutActMujocoUR5eParticle(RolloutAct, RolloutMujocoUR5eParticle):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eParticle()
    rollout.run()
