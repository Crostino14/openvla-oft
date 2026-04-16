"""
run_finetune_libero_goal.py

Orchestration script for fine-tuning OpenVLA-OFT on LIBERO Goal tasks.
Handles:
  - Downloading the base openvla-7b model from HuggingFace Hub (first run)
  - Submitting SLURM training jobs iteratively
  - Merging LoRA weights between iterations
  - Auto-resuming from the latest checkpoint

Usage:
  cd /home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/openvla-oft/vla-scripts
  python run_finetune_libero_goal.py
"""

import os
import subprocess
import re
import time
import glob
import shutil
from huggingface_hub import snapshot_download


# ======================== CONFIGURATION ========================

# Base model
BASE_MODEL_NAME = "openvla-7b"
HF_REPO_ID = "openvla/openvla-7b"              # HuggingFace Hub repo for base model
DOWNLOAD_MODEL = True                           # Set to False if model is already downloaded

# Dataset
DATASET_NAME = "libero_goal_no_noops"
DATASET_FOLDER = "/mnt/beegfs/a.cardamone7/datasets/modified_libero_rlds"

# Checkpoints
RUN_ROOT_DIR = "/mnt/beegfs/a.cardamone7/checkpoints_saving_folder/openvla"

# Training config
USE_PROPRIO = True
LORA_RANK = 64

# Resume control
START_ITERATION = 1                             # Set > 1 to resume from a previous run
OLD_RESUME_STEP = -1                            # Set to -1 for fresh start, or last merged step
RESUME = False
RESUME_STEP = 0

# Run ID note (describes the architecture/config)
if USE_PROPRIO:
    RUN_ID_NOTE = f"parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio"
else:
    RUN_ID_NOTE = f"parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img"

# Number of training iterations (each iteration is one SLURM job)
MAX_ITERATIONS = 50

# ===============================================================


def get_model_run_name():
    """Construct the model run name that finetune.py generates."""
    return (
        f"{BASE_MODEL_NAME}+{DATASET_NAME}"
        f"+b8+lr-0.0005+lora-r{LORA_RANK}+dropout-0.0"
        f"--image_aug--{RUN_ID_NOTE}"
    )


def get_highest_epoch(root_dir, model_name):
    """Find the highest checkpoint step in the run directory."""
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} does not exist.")
        return None

    print(f"Searching for folders in {root_dir} with model name {model_name}")
    folders = glob.glob(os.path.join(root_dir, f"{model_name}--*_chkpt"))
    if not folders:
        print("No checkpoint folders found.")
        return None

    folders.sort(
        key=lambda x: int(re.search(r'--(\d+)_chkpt', x).group(1)),
        reverse=True,
    )
    highest_epoch_folder = folders[0]
    highest_epoch = highest_epoch_folder.split("--")[-1].split("_chkpt")[0]
    print(f"Highest epoch folder: {highest_epoch_folder}")
    return highest_epoch


def wait_for_job(job_id):
    """Poll SLURM until the given job completes."""
    print(f"Job {job_id} submitted. Waiting for completion...")
    while True:
        result = subprocess.run(
            ['squeue', '--job', job_id],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Error checking job status: {result.stderr}")
            return False
        if job_id not in result.stdout:
            print(f"Job {job_id} completed.")
            return True
        time.sleep(30)


def submit_slurm_job(command):
    """Submit a SLURM job and return the job ID."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error submitting job: {result.stderr}")
        return None

    job_id = None
    for line in result.stdout.split('\n'):
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            break

    return job_id


def run_merge_lora_weights(last_epoch, model_name, new_last_epoch):
    """Merge LoRA weights from the latest checkpoint into the base model."""
    if last_epoch == -1:
        base_checkpoint = os.path.join(RUN_ROOT_DIR, BASE_MODEL_NAME)
    else:
        base_checkpoint = os.path.join(RUN_ROOT_DIR, f"{model_name}--{last_epoch}_chkpt")

    new_checkpoint = os.path.join(RUN_ROOT_DIR, f"{model_name}--{new_last_epoch}_chkpt")

    print(f"Merging LoRA weights:")
    print(f"  Base: {base_checkpoint}")
    print(f"  LoRA: {new_checkpoint}")

    # Copy modeling_prismatic.py from base checkpoint to new checkpoint
    modeling_prismatic_src = os.path.join(base_checkpoint, "modeling_prismatic.py")
    if os.path.exists(modeling_prismatic_src):
        shutil.copyfile(
            modeling_prismatic_src,
            os.path.join(new_checkpoint, "modeling_prismatic.py"),
        )
        time.sleep(5)
    else:
        print(f"WARNING: modeling_prismatic.py not found at {modeling_prismatic_src}")

    # Submit merge job
    job_id = submit_slurm_job([
        'sbatch', 'merge_lora_weights_and_save.sh',
        base_checkpoint, new_checkpoint,
    ])
    if job_id is None:
        return False

    return wait_for_job(job_id)


def download_base_model():
    """Download the base openvla-7b model from HuggingFace Hub."""
    model_dir = os.path.join(RUN_ROOT_DIR, BASE_MODEL_NAME)
    
    # Check if model already exists (complete download)
    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    config_file = os.path.join(model_dir, "config.json")
    if safetensors_files and os.path.exists(config_file):
        print(f"Base model already exists at {model_dir} ({len(safetensors_files)} safetensors files found).")
        return True

    print("Using local temp directory to avoid BeeGFS lock issues...")
    
    # Use local /tmp directory for download to avoid BeeGFS lock issues
    temp_dir = f"/tmp/{os.environ.get('USER', 'user')}_openvla_download"
    
    # Clean up any previous temp download
    if os.path.exists(temp_dir):
        print(f"Removing previous temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Download to local temp directory first
        print(f"Downloading to temporary location: {temp_dir}")
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=temp_dir,
        )
        
        # Verify download
        safetensors_files = glob.glob(os.path.join(temp_dir, "*.safetensors"))
        if not safetensors_files:
            print("ERROR: No safetensors files found after download!")
            return False
        
        print(f"Download completed! Found {len(safetensors_files)} safetensors files.")
        
        # Now copy to BeeGFS destination
        print(f"Copying model to BeeGFS: {model_dir}")
        if os.path.exists(model_dir):
            print("Removing partial BeeGFS directory...")
            shutil.rmtree(model_dir)
        
        shutil.copytree(temp_dir, model_dir)
        print("Copy to BeeGFS completed!")
        
        # Clean up temp directory
        print("Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
        
        # Clear HF cache to save space
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                print("Clearing HuggingFace cache...")
                shutil.rmtree(cache_dir)
        except Exception as e:
            print(f"Note: Could not clear HF cache: {e}")

        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("="*60)
        print("Run these commands to download manually:")
        print(f"  cd {RUN_ROOT_DIR}")
        print(f"  rm -rf {BASE_MODEL_NAME}")
        print(f"  huggingface-cli download {HF_REPO_ID} --local-dir {BASE_MODEL_NAME}")
        print("="*60)
        return False


def run_training():
    """Main training loop: iteratively submits finetune + merge jobs."""
    global RESUME, RESUME_STEP

    # Ensure output directory exists
    os.makedirs(RUN_ROOT_DIR, exist_ok=True)

    # Download base model if needed
    if DOWNLOAD_MODEL and START_ITERATION == 1:
        if not download_base_model():
            print("Failed to download base model. Exiting.")
            return

    old_resume_step = OLD_RESUME_STEP
    ckpt_folder = BASE_MODEL_NAME
    model_name = get_model_run_name()

    for i in range(START_ITERATION, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"  ITERATION {i}")
        print(f"{'='*60}")

        if i != 1:
            # Find the latest checkpoint and merge LoRA weights
            new_resume_step = get_highest_epoch(RUN_ROOT_DIR, model_name)
            if new_resume_step is None:
                print("No checkpoints found. Cannot resume. Exiting.")
                return

            if old_resume_step == new_resume_step:
                print(f"Resume step unchanged ({old_resume_step}). Skipping merge.")
            else:
                print(f"New highest epoch: {new_resume_step}")
                success = run_merge_lora_weights(old_resume_step, model_name, new_resume_step)
                if not success:
                    print("LoRA merge failed. Exiting.")
                    return
                old_resume_step = new_resume_step

            RESUME_STEP = new_resume_step if new_resume_step else old_resume_step
            RESUME = True
            ckpt_folder = f"{model_name}--{RESUME_STEP}_chkpt"

        # Build SLURM arguments
        bash_args = [
            f"{ckpt_folder}",
            f"{RUN_ID_NOTE}",
            f"{RESUME}",
            f"{RESUME_STEP}",
            f"{RUN_ROOT_DIR}",
            f"{DATASET_NAME}",
            f"{USE_PROPRIO}",
            f"{DATASET_FOLDER}",
        ]

        print(f"Submitting training job with args: {bash_args}")
        job_id = submit_slurm_job(['sbatch', 'finetune_libero_goal.sh'] + bash_args)
        if job_id is None:
            print("Failed to submit training job. Exiting.")
            return

        success = wait_for_job(job_id)
        if not success:
            print("Training job failed. Exiting.")
            return

    print(f"\nAll {MAX_ITERATIONS} iterations completed successfully!")


if __name__ == "__main__":
    run_training()
