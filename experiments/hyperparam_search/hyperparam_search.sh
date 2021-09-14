#!/bin/bash
set -e

cd fine_tune_tag-frozen
bash hyperparam_search.sh

cd ../fine_tune_tag
bash hyperparam_search.sh

cd ../lang_grad_rev_tag
bash hyperparam_search.sh

cd ../lang_ent_max_tag
bash hyperparam_search.sh

cd ../fine_tune_cls-frozen
bash hyperparam_search.sh

cd ../fine_tune_cls
bash hyperparam_search.sh

cd ../lang_grad_rev_cls
bash hyperparam_search.sh

cd ../lang_ent_max_cls
bash hyperparam_search.sh
