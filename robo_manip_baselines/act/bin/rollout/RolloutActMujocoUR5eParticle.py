from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoUR5eParticle


class RolloutActMujocoUR5eParticle(RolloutAct, RolloutMujocoUR5eParticle):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoUR5eParticle()
    rollout.run()
