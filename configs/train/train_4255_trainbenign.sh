#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
# eval "$(conda shell.bash hook)"
# conda activate dire
EXP_NAME="imagenet-pgd_4255-trainbenign-adm-uncond"
DATASETS="pgd_4255_trainbenign"
DATASETS_TEST="pgd_4255"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST