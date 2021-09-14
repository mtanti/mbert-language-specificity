source ~/data/mufins/mufins-project/venv_mufins/bin/activate

python ~/data/mufins/mufins-project/bin/dataprocs/udpos/preprocess.py \
    --src_path ~/data/mufins/datasets/udpos/raw \
    --dst_path . \
    --tokeniser_name "mbert" \
    --train_fraction 0.9 \
    --val_fraction 0.1 \
    --max_num_tokens 128 \
    --seed 0 \
    --verbose yes \
    --debug_mode yes
