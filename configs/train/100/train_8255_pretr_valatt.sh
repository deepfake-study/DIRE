#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
# eval "$(conda shell.bash hook)"
# conda activate dire
EXP_NAME="imagenet-pgd-100-8255-adm-uncond-valatt"
DATASETS="pgd_8255_100_valatt"
DATASETS_TEST="pgd_8255_100_valatt"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST pretrained True batch_size 32