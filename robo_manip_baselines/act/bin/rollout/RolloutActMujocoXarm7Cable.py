from robo_manip_baselines.act import RolloutAct
from robo_manip_baselines.common.rollout import RolloutMujocoXarm7Cable


class RolloutActMujocoXarm7Cable(RolloutAct, RolloutMujocoXarm7Cable):
    pass


if __name__ == "__main__":
    rollout = RolloutActMujocoXarm7Cable()
    rollout.run()
