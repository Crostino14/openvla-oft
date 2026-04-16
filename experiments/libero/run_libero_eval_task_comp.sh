#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=L2_OpenVLA_task_comp_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1          # 3 seeds x 2 parts = 6 jobs
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/eval_openvla_50000_task_comp_arr%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/eval_openvla_50000_task_comp_arr%a_%j.err

# ==========================================
# OpenVLA-OFT - Task Composition Evaluation
# ==========================================
# Supports both L1 and L2 composition levels.
#
# Usage:
#   COMP_LEVEL=l1 sbatch run_libero_eval_task_comp.sh
#   COMP_LEVEL=l2 sbatch run_libero_eval_task_comp.sh
#
# L1: Task-level generalization (new object-target pairs, 5 tasks)
# L2: Multi-step compositional generalization (chaining primitives, 5 tasks)
#
# Split: 5 tasks across 2 nodes per seed
#   Part 0 → tasks 0,1,2  (3 tasks)
#   Part 1 → tasks 3,4    (2 tasks)
# Array mapping: index = seed * 2 + part
#   0 → seed 0, part 0   |   1 → seed 0, part 1
#   2 → seed 1, part 0   |   3 → seed 1, part 1
#   4 → seed 2, part 0   |   5 → seed 2, part 1
# ==========================================

# Default to l1 if not set
COMP_LEVEL=${COMP_LEVEL:-l2}

# Decode seed and part from array index
SEED=$((SLURM_ARRAY_TASK_ID / 2))
PART=$((SLURM_ARRAY_TASK_ID % 2))

if [ $PART -eq 0 ]; then
    TASK_START=0
    TASK_END=3
else
    TASK_START=3
    TASK_END=5
fi

ID_NOTE="openvla_task_comp_${COMP_LEVEL}_seed_${SEED}_part${PART}"

# Model configuration
MODEL_PATH="/home/A.CARDAMONE7/checkpoints/openvla-7b-oft-libero-goal-seed${SEED}"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

echo "=========================================="
echo "OpenVLA-OFT Task Composition ${COMP_LEVEL^^} Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Comp Level: $COMP_LEVEL"
echo "Seed: $SEED"
echo "Part: $PART (tasks $TASK_START to $((TASK_END - 1)))"
echo "Model: $MODEL_PATH"
echo "Start time: $(date)"
echo ""

# ==========================================
# Environment Setup
# ==========================================

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1

# ==========================================
# Activate Conda Environment
# ==========================================

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

echo "Working directory: ${WORK_DIR}"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ==========================================
# Run Evaluation
# ==========================================

cd ${WORK_DIR}

python run_libero_eval_task_comp.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --model_family openvla \
    --task_suite_name libero_goal \
    --unnorm_key libero_goal_noops \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True \
    --num_open_loop_steps 8 \
    --center_crop True \
    --num_trials_per_task 50 \
    --env_img_res 256 \
    --comp_level ${COMP_LEVEL} \
    --task_start ${TASK_START} \
    --task_end ${TASK_END} \
    --seed ${SEED} \
    --run_id_note ${ID_NOTE} \
    --local_log_dir /mnt/beegfs/a.cardamone7/outputs/logs

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task Composition ${COMP_LEVEL^^} evaluation completed successfully!"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "Comp Level: $COMP_LEVEL"
echo "Seed: $SEED"
echo "Finish time: $(date)"
echo "=========================================="

exit $EXIT_CODE
