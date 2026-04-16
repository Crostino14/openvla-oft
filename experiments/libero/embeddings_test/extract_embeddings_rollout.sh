#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=extract_emb_rollout
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4 
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:59:00
#SBATCH --array=2
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/extract_emb_rollout_%A_%a.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/extract_emb_rollout_%A_%a.err

# Configuration
CHECKPOINT_PATH="/home/A.CARDAMONE7/checkpoints/openvla-7b-oft-libero-goal"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"
OUTPUT_DIR="/mnt/beegfs/a.cardamone7/outputs/embeddings/openvla/l2_first_step_only"

# Command levels array (index 0-3)
COMMAND_LEVELS=("default" "l1" "l2" "l3")
COMMAND_LEVEL=${COMMAND_LEVELS[$SLURM_ARRAY_TASK_ID]}

# Rollout parameters
TASK_SUITE="libero_goal"
NUM_ROLLOUTS=10  # 10 rollouts per task (10 tasks = 100 episodes per command level)
FIRST_STEP_ONLY=true  # Set to false for full rollout

echo "=========================================="
echo "Extracting Embeddings During Real Rollout"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID (Array: $SLURM_ARRAY_TASK_ID)"
echo "Start time: $(date)"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Task suite: $TASK_SUITE"
echo "Command level: $COMMAND_LEVEL"
echo "Rollouts per task: $NUM_ROLLOUTS"
echo "First step only: $FIRST_STEP_ONLY"
echo "Total episodes: $((NUM_ROLLOUTS * 10))"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Setup environment
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Activate conda
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

# Change to work directory
cd ${WORK_DIR}

# Extract embeddings during real inference rollouts
FIRST_STEP_FLAG=""
MODE_SUFFIX="full"
if [ "$FIRST_STEP_ONLY" = true ]; then
    FIRST_STEP_FLAG="--first_step_only"
    MODE_SUFFIX="first_step"
fi

python extract_embeddings_rollout.py \
    --checkpoint ${CHECKPOINT_PATH} \
    --task_suite ${TASK_SUITE} \
    --command_levels ${COMMAND_LEVEL} \
    --output_dir ${OUTPUT_DIR} \
    --num_rollouts ${NUM_ROLLOUTS} \
    ${FIRST_STEP_FLAG}

echo ""
echo "=========================================="
echo "Finish time: $(date)"
echo "Embedding extraction completed!"
echo "Command level: ${COMMAND_LEVEL}"
echo "Mode: ${MODE_SUFFIX}"
echo "Output saved to: ${OUTPUT_DIR}/rollout_embeddings_${TASK_SUITE}_${COMMAND_LEVEL}_${MODE_SUFFIX}_r${NUM_ROLLOUTS}.pkl"
echo "=========================================="
