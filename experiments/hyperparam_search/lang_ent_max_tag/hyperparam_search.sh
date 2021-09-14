source ~/data/mufins/mufins-project/venv_mufins/bin/activate

python ~/data/mufins/mufins-project/bin/random_parameter_space_generator.py \
    --spec_file_path hyperparam_specs.json \
    --output_file_path parameter_space.txt \
    --amount 20 \
    --seed 0

python ~/data/mufins/mufins-project/bin/experiments/lang_ent_max_tag/experiment.py \
    --label_src_path ~/data/mufins/datasets/udpos/processed/udpos-2/ \
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
    --default_lang_error_weighting 0.5 \
    --parameter_space_path parameter_space.txt \
    --hyperparameter_search_mode "yes" \
    --batch_size 32 \
    --seed 0 \
    --verbose yes \
    --debug_mode yes
