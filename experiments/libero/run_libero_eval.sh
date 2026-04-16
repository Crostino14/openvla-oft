#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=task10_l2v_openvla_libero_eval
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode13
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=1 # Esegue 3 job (seed 0, 1, 2)
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/L2_Variations_Eval_openvla_libero_goal_task10_seed_%a_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/L2_Variations_Eval_openvla_libero_goal_task10_seed_%a_%j.err

SEED=$SLURM_ARRAY_TASK_ID
ID_NOTE="L2_Variations_Eval_openvla_libero_goal_task10_seed_${SEED}"

# Model configuration
MODEL_PATH="/mnt/beegfs/a.cardamone7/checkpoints/openvla-7b-oft-libero-goal-seed${SEED}"
WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

echo "=========================================="
echo "Starting LIBERO Goal Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID (Seed): $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Model: $MODEL_PATH"
echo "Start time: $(date)"
echo ""

# Setup environment
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS##*:}
export CUDA_LAUNCH_BLOCKING=1

# Activate conda
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

# Change to work directory
cd ${WORK_DIR}

# Run evaluation
python run_libero_eval.py \
    --pretrained_checkpoint ${MODEL_PATH} \
    --model_family openvla \
    --task_suite_name libero_goal \
    --unnorm_key libero_goal_no_noops \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_l1_regression True \
    --num_open_loop_steps 8 \
    --center_crop True \
    --num_trials_per_task 50 \
    --env_img_res 256 \
    --seed ${SEED} \
    --change_command True \
    --command_level l2 \
    --task_id 9 \
    --only_numbered_variants True \
    --run_id_note ${ID_NOTE} \
    --local_log_dir /mnt/beegfs/a.cardamone7/outputs/logs

echo ""
echo "Finish time: $(date)"
echo "=========================================="
echo "Evaluation with seed $SEED completed!"