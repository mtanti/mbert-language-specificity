source ~/data/mufins/mufins-project/venv_mufins/bin/activate

python ~/data/mufins/mufins-project/bin/experiments/lang_grad_rev_cls/experiment.py \
    --label_src_path ~/data/mufins/datasets/xnli/processed/xnli-5/ \
    --lang_src_path ~/data/mufins/datasets/wikipedia/processed/wikipedia-2/ \
    --dst_path . \
    --device_name "cuda:1" \
    --default_encoder_name "mbert" \
    --default_init_stddev "1e-1" \
    --default_minibatch_size 64 \
    --default_dropout_rate "0.1" \
    --default_freeze_embeddings "no" \
    --default_encoder_learning_rate "2e-5" \
    --default_postencoder_learning_rate "1e-2" \
    --default_max_epochs 5 \
    --default_grl_lambda 0.0 \
    --parameter_space_path parameter_space.txt \
    --hyperparameter_search_mode "no" \
    --batch_size 32 \
    --seed 0 \
    --verbose yes \
    --debug_mode yes
