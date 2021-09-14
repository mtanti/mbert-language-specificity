source ~/data/mufins/mufins-project/venv_mufins/bin/activate

python ~/data/mufins/mufins-project/bin/dataprocs/wikipedia/preprocess.py \
    --src_path ~/data/mufins/datasets/wikipedia/raw \
    --dst_path . \
    --tokeniser_name "mbert" \
    --train_fraction 0.7 \
    --val_fraction 0.1 \
    --dev_fraction 0.1 \
    --test_fraction 0.1 \
    --max_num_texts_per_lang 5000 \
    --min_num_chars 100 \
    --max_num_tokens 128 \
    --seed 0 \
    --verbose yes \
    --debug_mode yes
