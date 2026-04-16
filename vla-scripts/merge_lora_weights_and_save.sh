#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --export=ALL

BASE_CHECKPOINT=$1
LORA_FINETUNED_CHECKPOINT_DIR=$2
srun python merge_lora_weights_and_save.py \
    --base_checkpoint $BASE_CHECKPOINT \
    --lora_finetuned_checkpoint_dir $LORA_FINETUNED_CHECKPOINT_DIR
