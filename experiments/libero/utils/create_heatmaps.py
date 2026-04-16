import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import debugpy
import glob
import pickle as pkl

TABLE_SIZE = (1.0, 1.0)  # height (y), width (x)


def infer_command_level_from_path(path):
    """Infer command variation level from a rollout path."""
    normalized_path = path.lower()
    if "command_l1" in normalized_path:
        return "L1"
    if "command_l2" in normalized_path:
        return "L2"
    if "command_l3" in normalized_path:
        return "L3"
    if "ablation" in normalized_path:
        return "ABLATION"
    if "task_comp_l1" in normalized_path:
        return "L1"
    if "task_comp_l2" in normalized_path:
        return "L2"
    return "DEFAULT"


def get_rollout_paths(base_path, model_prefix):
    """Return rollout folders for direct or multi-config layouts."""
    run_folders_direct = glob.glob(os.path.join(base_path, "run_*"))
    if run_folders_direct:
        return [(base_path, model_prefix)]

    normalized_path = base_path.lower()
    if "task_comp" in normalized_path or "task_composition" in normalized_path:
        return [
            (os.path.join(base_path, "default"), f"{model_prefix} (Default)"),
            (os.path.join(base_path, "command_l1"), f"{model_prefix} (L1)"),
            (os.path.join(base_path, "command_l2"), f"{model_prefix} (L2)"),
            (os.path.join(base_path, "command_ablation"), f"{model_prefix} (Ablation)"),
        ]

    return [
        (os.path.join(base_path, "default"), f"{model_prefix} (Default)"),
        (os.path.join(base_path, "command_l1"), f"{model_prefix} (L1)"),
        (os.path.join(base_path, "command_l2"), f"{model_prefix} (L2)"),
        (os.path.join(base_path, "command_l3"), f"{model_prefix} (L3)"),
        (os.path.join(base_path, "command_ablation"), f"{model_prefix} (Ablation)"),
    ]

def compute_heatmap_data(task_distribution):
    """Compute heatmap data from trajectories (without plotting)."""
    px_resolution = 0.5  # in cm
    table_size_cm = np.array(TABLE_SIZE) * 100
    table_size_px = (table_size_cm / px_resolution).astype(np.int32)
    table_map = np.zeros((table_size_px[0], table_size_px[1]))

    for episode_idx, trajectory in enumerate(task_distribution):
        trajectory = np.array(trajectory)[:, :2]
        px_traj = (trajectory * 100 / px_resolution).astype(np.int32)
        px_traj[:, 0] = table_map.shape[0] // 2 + px_traj[:, 0]
        px_traj[:, 1] = table_map.shape[1] // 2 + px_traj[:, 1]
        px_traj = px_traj[
            (px_traj[:, 0] >= 0) & (px_traj[:, 0] < table_map.shape[0]) &
            (px_traj[:, 1] >= 0) & (px_traj[:, 1] < table_map.shape[1])
        ]
        for x, y in px_traj:
            table_map[x, y] += 1

    # Crop to focus area
    y_min, y_max = -45, 20
    x_min, x_max = -35, 35
    y_min_px = int((y_min + table_size_cm[0] / 2) / px_resolution)
    y_max_px = int((y_max + table_size_cm[0] / 2) / px_resolution)
    x_min_px = int((x_min + table_size_cm[1] / 2) / px_resolution)
    x_max_px = int((x_max + table_size_cm[1] / 2) / px_resolution)
    cropped_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]
    
    return cropped_map


def heat_map(task_distribution, task_path, task_name, model_name, command_level="DEFAULT"):
    # Each px represents 0.5x0.5 cm (0.005m x 0.005m)
    px_resolution = 0.5  # in cm

    # Convert TABLE_SIZE to cm and then to pixels
    table_size_cm = np.array(TABLE_SIZE) * 100  # meters to cm
    table_size_px = (table_size_cm / px_resolution).astype(np.int32)

    # Initialize table heatmap
    table_map = np.zeros((table_size_px[0], table_size_px[1]))  # shape: (height, width)

    # Loop through each episode's trajectory
    for episode_idx, trajectory in enumerate(task_distribution):
        print(f"Processing episode: {episode_idx}")

        # Convert list of [x, y, z] to np.array and take only x, y
        trajectory = np.array(trajectory)[:, :2]  # shape: (T, 2)

        # Convert meters to cm and to pixels
        px_traj = (trajectory * 100 / px_resolution).astype(np.int32)

        # Translate coords: center of table is the middle of the image
        px_traj[:, 0] = table_map.shape[0] // 2 + px_traj[:, 0]  # x -> vertical axis (rows)
        px_traj[:, 1] = table_map.shape[1] // 2 + px_traj[:, 1]  # y -> horizontal axis (cols)

        # Clip to table bounds
        px_traj = px_traj[
            (px_traj[:, 0] >= 0) & (px_traj[:, 0] < table_map.shape[0]) &
            (px_traj[:, 1] >= 0) & (px_traj[:, 1] < table_map.shape[1])
        ]

        # Populate heatmap
        for x, y in px_traj:
            table_map[x, y] += 1

    # Set crop range in cm for visual focus (adjust as needed)
    y_min, y_max = -45, 20
    x_min, x_max = -35, 35
    task_title = task_name.replace("_", " ").title()

    # Convert to pixel bounds
    y_min_px = int((y_min + table_size_cm[0] / 2) / px_resolution)
    y_max_px = int((y_max + table_size_cm[0] / 2) / px_resolution)
    x_min_px = int((x_min + table_size_cm[1] / 2) / px_resolution)
    x_max_px = int((x_max + table_size_cm[1] / 2) / px_resolution)

    # Crop table
    cropped_map = table_map[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 14))
    plt.title(f"[{command_level.upper()}] \"{task_title}\"")
    plt.xlabel("Y Axis (cm)")
    plt.ylabel("X Axis (cm)")

    norm = mcolors.LogNorm(vmin=1, vmax=np.max(cropped_map) if np.max(cropped_map) > 0 else 1)
    im = ax.imshow(cropped_map, cmap='plasma', origin='upper', norm=norm)
    ax.invert_xaxis()  # Invert x-axis to match the coordinate system
    
    # Axis ticks (every 10 cm)
    ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
    ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
    tick_labels_x = np.arange(x_min, x_max, 10)
    tick_labels_y = np.arange(y_min, y_max, 10)

    plt.xticks(ticks_x, tick_labels_x)
    plt.yticks(ticks_y, tick_labels_y)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Trajectory Density (log scale)")

    # Save
    os.makedirs(task_path, exist_ok=True)
    save_path = os.path.join(task_path, f"{task_name}_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {save_path}")
    return cropped_map


def create_combined_heatmap(heatmaps_data, task_path, model_name, command_level):
    """
    Create a combined horizontal image with all heatmaps side by side.
    
    Args:
        heatmaps_data: dict of {task_name: cropped_map}
        task_path: path to save the combined image
        model_name: model name for the title (e.g., "OpenVLA-OFT")
        command_level: command level (e.g., "L1", "L2")
    """
    if not heatmaps_data:
        print("WARNING: No heatmaps to combine")
        return
    
    n_tasks = len(heatmaps_data)
    task_names = list(heatmaps_data.keys())
    
    # Calculate global vmax for consistent color scaling
    global_vmax = max(np.max(hm) for hm in heatmaps_data.values() if np.max(hm) > 0)
    global_vmax = max(global_vmax, 1)
    
    # Create figure with subplots
    fig_width = 5 * n_tasks + 1  # 5 inches per task + colorbar space
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, 8))
    
    if n_tasks == 1:
        axes = [axes]
    

    norm = mcolors.LogNorm(vmin=1, vmax=global_vmax)
    
    # Crop range for axis labels
    y_min, y_max = -45, 20
    x_min, x_max = -35, 35
    px_resolution = 0.5
    
    for idx, (ax, task_name) in enumerate(zip(axes, task_names)):
        cropped_map = heatmaps_data[task_name]
        task_title = task_name.replace("_", " ").title()
        
        im = ax.imshow(cropped_map, cmap='plasma', origin='upper', norm=norm)
        ax.invert_xaxis()
        
        # Set title (task command)
        ax.set_title(f'"{task_title}"', fontsize=9, wrap=True)
        
        # Axis ticks (every 10 cm)
        ticks_x = np.arange(0, cropped_map.shape[1], int(10 / px_resolution))
        ticks_y = np.arange(0, cropped_map.shape[0], int(10 / px_resolution))
        tick_labels_x = np.arange(x_min, x_max, 10)
        tick_labels_y = np.arange(y_min, y_max, 10)
        
        ax.set_xticks(ticks_x)
        ax.set_xticklabels(tick_labels_x, fontsize=7)
        ax.set_yticks(ticks_y)
        ax.set_yticklabels(tick_labels_y, fontsize=7)
        
        if idx == 0:
            ax.set_ylabel("X Axis (cm)", fontsize=9)
        ax.set_xlabel("Y Axis (cm)", fontsize=9)
    
    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Trajectory Density (log scale)", fontsize=10)
    
    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.08, wspace=0.15)
    
    # Save combined image
    os.makedirs(task_path, exist_ok=True)
    save_path = os.path.join(task_path, f"combined_heatmaps_{command_level.lower()}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved combined heatmap to {save_path}")


def process_rollout_folder(test_path, model_name, dataset_config=None):
    """Process all rollouts in a given folder."""
    print(f"\n{'='*80}")
    print(f"Processing: {test_path}")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")
    
    command_level = infer_command_level_from_path(test_path)
    
    print(f"Command level: {command_level}\n")
    
    run_folders = glob.glob(os.path.join(test_path, "run_*"))
    
    if not run_folders:
        print(f"WARNING: No run_* folders found in {test_path}")
        return

    tasks_trajectories = {}

    for run in run_folders:
        trajectories_npy = glob.glob(os.path.join(run, "*.npy"))
        trajectories_npy.sort(key=lambda x: int(os.path.basename(x).split("episode=")[-1].split("--")[0]))

        for trajectory_npy in trajectories_npy:
            data = np.load(trajectory_npy, allow_pickle=True).item()
            print(f"Analyzing {trajectory_npy.split('/')[-1]}...")
    
            # Extract task name from filename (format: ...--task=task_name.npy)
            filename = os.path.basename(trajectory_npy)
            task_name = filename.split("--task=")[-1].replace(".npy", "")
            
            print(f"Task command: {task_name}")
            if task_name not in tasks_trajectories:
                tasks_trajectories[task_name] = []
            
            # Check if 'states' key exists
            if 'states' in data:
                states = np.array(data['states'])[:, :3]
            else:
                print(f"  WARNING: 'states' key not found in {filename}")
                print(f"  The .npy file was created without state information.")
                print(f"  Please re-run evaluation with updated run_libero_eval.py to save states.")
                print(f"  Skipping this file...")
                continue
            
            # Denormalize positions if dataset_config is provided
            if dataset_config is not None:
                qpos_mean = dataset_config['qpos_mean'][:3]
                qpos_std = dataset_config['qpos_std'][:3]
                states = (states * qpos_std) + qpos_mean
            
            tasks_trajectories[task_name].append(states)  # list of [x, y, z]
    
    # Create heatmaps for each task
    if tasks_trajectories:
        heatmaps_data = {}
        for task_name, episodes in tasks_trajectories.items():
            print(f"Creating heatmap for task: {task_name} with {len(episodes)} episodes")
            cropped_map = heat_map(episodes, test_path, task_name, model_name, command_level)
            heatmaps_data[task_name] = cropped_map
        
        # Create combined heatmap
        print(f"\nCreating combined heatmap...")
        create_combined_heatmap(heatmaps_data, test_path, model_name, command_level)
    else:
        print(f"WARNING: No trajectories found in {test_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create heatmaps for OpenVLA test trajectories")
    argparser.add_argument('--debug', action='store_true', help="Enable remote debugging")
    argparser.add_argument('--dataset_config', type=str, default=None, 
                          help="Path to dataset_stats.pkl for denormalization (optional)")
    argparser.add_argument('--base_path', type=str, 
                          default="/home/A.CARDAMONE7/outputs/rollouts/libero_goal/syntactic_variation/openvla-oft/openvla-oft_20000_test/command_l2",
                          help="Base path for rollouts")
    args = argparser.parse_args()
    
    if args.debug:
        # Enable remote debugging
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    
    # Load dataset config if provided
    dataset_config = None
    if args.dataset_config and os.path.exists(args.dataset_config):
        print(f"Loading dataset config from: {args.dataset_config}")
        dataset_config = pkl.load(open(args.dataset_config, "rb"))
    else:
        print("No dataset config provided - assuming states are already denormalized")
    
    for path, model_name in get_rollout_paths(args.base_path, "OpenVLA-OFT"):
        if os.path.exists(path):
            process_rollout_folder(path, model_name, dataset_config)
        else:
            print(f"WARNING: Path does not exist: {path}")
    
    print("\n" + "="*80)
    print("HEATMAP GENERATION COMPLETE")
    print("="*80)
