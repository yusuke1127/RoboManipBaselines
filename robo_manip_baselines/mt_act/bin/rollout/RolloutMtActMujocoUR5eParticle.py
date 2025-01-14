from robo_manip_baselines.common.rollout import RolloutMujocoUR5eParticle
from robo_manip_baselines.mt_act import RolloutMtAct


class RolloutMtActMujocoUR5eParticle(RolloutMtAct, RolloutMujocoUR5eParticle):
    pass


if __name__ == "__main__":
    rollout = RolloutMtActMujocoUR5eParticle()
    rollout.run()
