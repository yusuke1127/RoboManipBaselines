skip=2
train_ratio=0.8
python ../utils/make_multi_dataset.py \
--in_dir ./data/ \
--out_dir ./data/skip_$skip/train_ratio_`echo $train_ratio|tr -d .` \
--skip $skip \
--train_ratio $train_ratio \
--nproc `nproc`
