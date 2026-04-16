#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=verify_visual_emb
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/verify_visual_emb_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/verify_visual_emb_%j.err

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

echo "=========================================="
echo "Verify Visual Embedding Independence"
echo "=========================================="
echo "Start time: $(date)"

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=${LIBERO_PATH}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

cd ${WORK_DIR}

python verify_visual_embedding.py

echo ""
echo "Finish time: $(date)"
echo "=========================================="
