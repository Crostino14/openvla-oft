#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=finetune_openvla_libero_goal
#SBATCH --partition=gpuq
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Only ONE task per node!
#SBATCH --gres=gpu:4                  # Request 4 GPUs per node
#SBATCH --cpus-per-task=64            # Adjust for data loading, etc.
#SBATCH --exclude=gnode13             # Exclude gnode13 - it has busy GPUs
#SBATCH --exclusive
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/finetune_openvla_libero_goal_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/finetune_openvla_libero_goal_%j.err

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export WANDB_MODE=offline

# Add openvla-oft root to PYTHONPATH
OPENVLA_ROOT="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft"
export PYTHONPATH="${OPENVLA_ROOT}:${PYTHONPATH}"

mkdir -p /tmp/$USER/triton_cache
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache

# --- Parameters (with defaults) ---
CKPT_FOLDER="${1:-openvla-7b}"
RUN_ID_NOTE="${2:-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio}"
RESUME="${3:-true}"
RESUME_STEP="${4:-0}"
RUN_ROOT_DIR="${5:-/mnt/beegfs/a.cardamone7/checkpoints_saving_folder/openvla}"
DATASET_NAME="${6:-libero_goal_no_noops}"
USE_PROPRIO="${7:-True}"
DATASET_FOLDER="${8:-/mnt/beegfs/a.cardamone7/datasets/modified_libero_rlds}"

echo "=========================================="
echo "OpenVLA-OFT Fine-tuning on LIBERO Goal"
echo "=========================================="
echo "Checkpoint folder: $CKPT_FOLDER"
echo "Run ID note: $RUN_ID_NOTE"
echo "Resume: $RESUME"
echo "Resume step: $RESUME_STEP"
echo "Run root directory: $RUN_ROOT_DIR"
echo "Dataset name: $DATASET_NAME"
echo "Use proprioception: $USE_PROPRIO"
echo "Dataset folder: $DATASET_FOLDER"
echo "=========================================="

# Assign a unique port based on the Slurm job ID
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

torchrun --standalone --nnodes 1 --nproc-per-node 4 --master-port $MASTER_PORT finetune.py \
    --vla_path ${RUN_ROOT_DIR}/${CKPT_FOLDER} \
    --data_root_dir ${DATASET_FOLDER} \
    --dataset_name $DATASET_NAME \
    --run_root_dir $RUN_ROOT_DIR \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio ${USE_PROPRIO} \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 50000 \
    --save_freq 500 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 64 \
    --wandb_entity "openvla_libero_goal" \
    --wandb_project "OpenVLA_OFT_finetune_libero_goal" \
    --run_id_note $RUN_ID_NOTE \
    --resume $RESUME \
    --resume_step $RESUME_STEP
