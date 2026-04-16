"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import time

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, change_command=False, command_level=None, ablation_bddl_file=None, resolution=256):
    """
    Initializes and returns the LIBERO environment, along with the task descriptions.
    
    Args:
        task: LIBERO task object
        change_command: If True, uses modified BDDL files with synonym commands (L1/L2/L3)
        command_level: Level of command variation ('l1', 'l2', 'l3') or None for default
        ablation_bddl_file: Full BDDL filename for ablation (e.g., 'turn_on_the_stove_ablation_stove1.bddl')
        resolution: Camera resolution
    
    Returns:
        env: LIBERO environment
        task_description: str - Current command being used (original, variation, or ablation)
        original_description: str - Original command from default BDDL
    """
    # Get original BDDL file path
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    # Extract original command from default BDDL file
    original_description = extract_command_from_bddl(task_bddl_file)
    
    # Fallback to task.language if extraction fails
    if original_description is None:
        original_description = task.language
        print(f"Warning: Could not extract command from BDDL, using task.language: {original_description}")
    
    # Current task description starts as original
    task_description = original_description
    
    # PRIORITY 1: Handle ablation BDDL files (explicit filename)
    if ablation_bddl_file is not None:
        # Use explicit BDDL file path
        new_task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, ablation_bddl_file)
        
        if os.path.exists(new_task_bddl_file):
            print(f"✓ Using ablation BDDL: {new_task_bddl_file}")
            task_bddl_file = new_task_bddl_file
            
            # Extract command from ablation BDDL
            ablation_description = extract_command_from_bddl(task_bddl_file)
            if ablation_description:
                task_description = ablation_description
                print(f"✓ Ablation command from BDDL: '{task_description}'")
            else:
                print(f"Warning: Could not extract command from ablation BDDL")
        else:
            print(f"ERROR: Ablation BDDL file not found: {new_task_bddl_file}")
            print(f"Falling back to default BDDL file: {task_bddl_file}")
    
    # PRIORITY 2: Handle L1/L2/L3 synonym variations (if ablation_bddl_file is None)
    elif change_command and command_level is not None:
        # Replace .bddl with _syn_l{X}_vN.bddl
        base_name = task.bddl_file.replace('.bddl', '')
        bddl_folder = os.path.join(get_libero_path("bddl_files"), task.problem_folder)
        
        # Pattern: {base_name}_syn_{command_level}_v*.bddl (case-insensitive)
        pattern = f"{base_name}_syn_{command_level}_v"
        
        # Find all matching files
        matching_files = []
        try:
            for filename in os.listdir(bddl_folder):
                if pattern.lower() in filename.lower() and filename.endswith('.bddl'):
                    matching_files.append(filename)
        except Exception as e:
            print(f"Warning: Could not list files in {bddl_folder}: {e}")
        
        if matching_files:
            # Extract version numbers and sort to get the highest version
            def extract_version(filename):
                import re
                match = re.search(r'_v(\d+)', filename, re.IGNORECASE)
                return int(match.group(1)) if match else 0
            
            matching_files.sort(key=extract_version, reverse=True)
            selected_file = matching_files[0]
            new_task_bddl_file = os.path.join(bddl_folder, selected_file)
            
            print(f"✓ Found {len(matching_files)} matching file(s) for {command_level}")
            print(f"✓ Using version file: {selected_file}")
            task_bddl_file = new_task_bddl_file
            
            # Extract variation command from custom BDDL
            variation_description = extract_command_from_bddl(task_bddl_file)
            if variation_description:
                task_description = variation_description
                print(f"✓ Command loaded: '{task_description}'")
            else:
                print(f"Warning: Could not extract variation command from {selected_file}")
        else:
            # Fallback: try without version suffix
            new_bddl_filename = f"{base_name}_syn_{command_level}.bddl"
            new_task_bddl_file = os.path.join(bddl_folder, new_bddl_filename)
            
            if os.path.exists(new_task_bddl_file):
                print(f"✓ Using custom BDDL (no version): {new_bddl_filename}")
                task_bddl_file = new_task_bddl_file
                variation_description = extract_command_from_bddl(task_bddl_file)
                if variation_description:
                    task_description = variation_description
            else:
                print(f"WARNING: No custom BDDL files found for {command_level}")
                print(f"Falling back to default BDDL file: {task_bddl_file}")
    
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    
    return env, task_description, original_description

def extract_command_from_bddl(bddl_file_path):
    """
    Extract task command/description from a BDDL file.
    
    Args:
        bddl_file_path: Path to the BDDL file
    
    Returns:
        str: Task command/description, or None if not found
    
    Example:
        From "(:language Open the middle layer of the drawer)"
        Returns "Open the middle layer of the drawer"
    """
    import re
    
    if not os.path.exists(bddl_file_path):
        return None
    
    try:
        with open(bddl_file_path, 'r') as f:
            content = f.read()
        
        # Pattern: (:language <comando>)
        # Estrae tutto tra ":language " e la parentesi chiusa
        pattern = r'\(:language\s+([^)]+)\)'
        
        match = re.search(pattern, content)
        if match:
            command = match.group(1).strip()
            return command
        
        return None
    
    except Exception as e:
        print(f"Warning: Could not parse BDDL file {bddl_file_path}: {e}")
        return None


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_traj, idx, success, task_description, log_file=None, dataset_name="libero_goal", run=0, change_command=False, command_level=None, custom_video_dir=None):
    """Saves an MP4 replay of an episode."""
    
    # Use custom directory if provided, otherwise use default logic
    if custom_video_dir is not None:
        rollout_dir = custom_video_dir
    else:
        # Original default behavior
        if change_command and command_level:
            rollout_dir = f"/mnt/beegfs/a.cardamone7/outputs/rollouts/{dataset_name}/syntactic_variation/openvla-oft/openvla-oft_50000_l1_variations_test/command_{command_level}/run_{run}"
        else:
            rollout_dir = f"/mnt/beegfs/a.cardamone7/outputs/rollouts/{dataset_name}/syntactic_variation/openvla-oft/openvla-oft_50000_l1_variations_test/default/run_{run}"
    
    os.makedirs(rollout_dir, exist_ok=True)
    
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_traj['image']:
        video_writer.append_data(img)
    video_writer.close()
    
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
        
    # save replay trajectory as a numpy file
    npy_path = mp4_path.replace('.mp4', '.npy')
    np.save(npy_path, rollout_traj)
    print(f"Saved replay trajectory at path {npy_path}")    
    
    return mp4_path

def quat2axisangle(quat):
    """
    Copied from robosuite: [https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55](https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55)

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den