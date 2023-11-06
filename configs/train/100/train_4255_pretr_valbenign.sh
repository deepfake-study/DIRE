#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
# eval "$(conda shell.bash hook)"
# conda activate dire
EXP_NAME="imagenet-pgd-100-4255-adm-uncond-valbenign"
DATASETS="pgd_4255_100_valbenign"
DATASETS_TEST="pgd_4255_100_valbenign"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST pretrained True batch_size 32