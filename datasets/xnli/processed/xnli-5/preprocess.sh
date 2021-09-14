source ~/data/mufins/mufins-project/venv_mufins/bin/activate

python ~/data/mufins/mufins-project/bin/dataprocs/xnli/preprocess.py \
    --src_path_xnli ~/data/mufins/datasets/xnli/raw/XNLI-1.0 \
    --src_path_multinli ~/data/mufins/datasets/xnli/raw/multinli_1.0 \
    --dst_path . \
    --tokeniser_name "mbert" \
    --train_fraction 0.9 \
    --val_fraction 0.1 \
    --max_num_tokens 128 \
    --seed 0 \
    --verbose yes \
    --debug_mode yes
