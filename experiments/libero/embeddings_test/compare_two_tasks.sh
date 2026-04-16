#!/bin/bash

#SBATCH --account=did_robot_learning_359
#SBATCH --job-name=compare_two_tasks
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/beegfs/a.cardamone7/outputs/logs/compare_two_tasks_openvla_l3_%j.out
#SBATCH --error=/mnt/beegfs/a.cardamone7/outputs/logs/compare_two_tasks_openvla_l3_%j.err

WORK_DIR="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/experiments/libero"
LIBERO_PATH="/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO"

echo "=========================================="
echo "Compare Two Tasks - OpenVLA-OFT"
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

# ── Task disponibili (LIBERO Goal) ──
#  0. put_the_wine_bottle_on_top_of_the_cabinet
#  1. open_the_top_drawer_and_put_the_bowl_inside
#  2. turn_on_the_stove
#  3. put_the_bowl_on_top_of_the_cabinet
#  4. put_the_bowl_on_the_plate
#  5. put_the_wine_bottle_on_the_rack
#  6. put_the_cream_cheese_in_the_bowl
#  7. open_the_middle_drawer_of_the_cabinet
#  8. push_the_plate_to_the_front_of_the_stove
#  9. put_the_bowl_on_the_stove
#
# Livelli di variazione: default, l1, l2, l3

# ── Modifica qui i parametri del confronto ──
TASK_A="put_the_wine_bottle_on_the_rack"
LEVEL_A="l3"

TASK_B="put_the_bowl_on_the_stove"
LEVEL_B="default"

# Decommentare per forzare la stessa scena visiva per entrambi i comandi:
SCENE_TASK="put_the_wine_bottle_on_the_rack"

# Decommentare per calcolare anche gli embedding pre-fusione (puramente testuali):
PRE_FUSION=true

CMD="python compare_two_tasks.py --task_a ${TASK_A} --level_a ${LEVEL_A} --task_b ${TASK_B} --level_b ${LEVEL_B}"

if [ -n "${SCENE_TASK}" ]; then
    CMD="${CMD} --scene_task ${SCENE_TASK}"
fi

if [ "${PRE_FUSION}" = "true" ]; then
    CMD="${CMD} --pre_fusion"
fi

echo "Comando: ${CMD}"
echo ""

${CMD}

echo ""
echo "Finish time: $(date)"
echo "=========================================="
