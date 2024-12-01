from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Ring


class RolloutActMujocoXarm7Ring(RolloutAct, RolloutMujocoXarm7Ring):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoXarm7Ring()
    rollout.run()
