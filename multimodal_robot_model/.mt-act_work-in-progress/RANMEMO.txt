## Install

Install [SARNN](../sarnn) according to [here](../sarnn/README.md).

cd multimodal_robot_model/mt_act/
git clone https://github.com/robopen/roboagent.git

python3 -m venv ./venv-mt-act
source ./venv-mt-act/bin/activate

pip install torchvision \
torch \
pyquaternion \
pyyaml \
rospkg \
pexpect \
mujoco \
dm_control \
opencv-python \
matplotlib \
einops \
packaging \
h5py \
h5py_cache
cd roboagent/detr && pip install -e .

pip install wandb \
ipython \
lru_cache

sed -i "s/mj_envs/robohive/g" ./evaluate.py


train_keywords="env1 env3 env5"
test_keywords="env0 env2 env4" 
python ../utils/make_dataset.py \
--in_dir ./data/ \
--out_dir ./data_tra-`echo "$train_keywords"|tr -d " "|tr -d "env"`_tes-`echo "$test_keywords"|tr -d " "|tr -d "env"` \
--train_keywords $train_keywords \
--test_keywords $test_keywords \
--nproc `nproc`

python ./roboagent/train.py --dataset_dir ./data/mnt/raid5/data/roboset/v0.4/baking_prep/ --ckpt_dir ./ckpt --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 6 --dim_feedforward 32 --seed 0 --temporal_agg --num_epochs 20 --lr 1e-5 --task_name pick_butter --run_name run_name

