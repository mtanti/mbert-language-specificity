#!/bin/bash
set -e

cd fine_tune_tag-frozen
bash experiment.sh

cd ../fine_tune_tag
bash experiment.sh

cd ../lang_grad_rev_tag
bash experiment.sh

cd ../lang_ent_max_tag
bash experiment.sh

cd ../fine_tune_cls-frozen
bash experiment.sh

cd ../fine_tune_cls
bash experiment.sh

cd ../lang_grad_rev_cls
bash experiment.sh

cd ../lang_ent_max_cls
bash experiment.sh
