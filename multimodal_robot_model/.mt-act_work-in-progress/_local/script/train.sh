skip=2
train_ratio=0.8
python ./roboagent/train.py \
--dataset_dir ./data/skip_$skip/train_ratio_`echo $train_ratio|tr -d .` \
--ckpt_dir     ./log/skip_$skip/train_ratio_`echo $train_ratio|tr -d .` \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 20 \
--hidden_dim 512 \
--batch_size 64 \
--dim_feedforward 3200 \
--seed 0 \
--temporal_agg \
--num_epochs 2000 \
--lr 1e-5 \
--multi_task \
--task_name pick_butter \
--run_name multi_task_run
