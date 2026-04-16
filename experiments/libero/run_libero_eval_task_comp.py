"""
run_libero_eval_task_comp.py
============================

Evaluates a trained Vision-Language-Action (VLA) policy on custom LIBERO
**task composition** scenarios, a generalization test benchmark distinct from
the standard LIBERO suite evaluation (``run_libero_eval.py``).

Scientific Motivation
---------------------
Task composition tests assess a form of *combinatorial generalization*:
the model is trained on a fixed set of primitive manipulation skills
(pick-place, push, open, insert) applied to specific object–target pairs,
then evaluated on **new combinations of those same primitives and objects
never seen together during training**. Success indicates that the policy has
factored the task representation rather than memorizing specific scene-action
associations.

This type of generalization is organized into two difficulty levels:

- **L1 (Object/Target Composition)**: Known manipulation primitives applied
  to new object–target pairings. For example, if the model learned
  ``cream_cheese → bowl`` and ``bowl → cabinet`` separately, L1 tests
  whether it can compose them to perform ``cream_cheese → cabinet``.

- **L2 (Temporal/Chain Composition)**: Sequential chains of multiple
  primitives that must be executed in order to complete a longer-horizon
  task, testing the model's ability to plan and complete multi-step
  behaviors beyond anything in the training set.

Task Definitions
----------------
All custom tasks share the ``libero_goal`` scene layout, which means they
can reuse initial simulator states recorded from corresponding libero_goal
training tasks (no new initial states need to be collected).

**L1 Tasks (object–target composition):**

1. ``put_the_plate_on_top_of_the_cabinet`` — trains: push_plate + bowl/bottle→cabinet
2. ``put_the_plate_on_the_stove`` — trains: push_plate + bowl→stove
3. ``put_the_cream_cheese_on_top_of_the_cabinet`` — trains: cheese→bowl + obj→cabinet
4. ``put_the_cream_cheese_on_the_plate`` — trains: cheese→bowl + bowl→plate
5. ``open_the_top_drawer_and_put_the_cream_cheese_inside`` — trains: open_drawer + bowl inside

**L2 Tasks (multi-step chains):**

1. ``open_the_middle_drawer_of_the_cabinet`` — chain: open + pick-place
2. ``put_the_bowl_on_the_stove`` — chain: pick-place + stove manipulation
3. ``put_the_cream_cheese_in_the_bowl`` — chain: 2 pick-place steps
4. ``push_the_plate_to_the_front_of_the_stove`` — chain: push + pick-place
5. ``put_the_bowl_on_top_of_the_cabinet`` — chain: 2 pick-place steps

Implementation Notes
--------------------
Unlike the standard evaluation script, this script does **not** use the
LIBERO benchmark registry (``benchmark.get_benchmark_dict()``) to find tasks.
Instead it:

1. Defines tasks explicitly in ``TASK_COMP_L1_TASKS`` / ``TASK_COMP_L2_TASKS``.
2. Builds ``Task`` NamedTuples pointing to custom ``.bddl`` files.
3. Borrows pre-recorded initial states (``.pruned_init``) from the closest
   matching libero_goal training task (same scene, matching object placement).
4. Instantiates each environment directly from its BDDL path via
   ``OffScreenRenderEnv`` rather than the benchmark factory.

This design allows the evaluation to run entirely in the libero_goal scene
without requiring new expert demonstrations or initial state collection.

Task Subset Support
-------------------
``cfg.task_start`` / ``cfg.task_end`` allow the evaluation workload to be
split across multiple compute nodes on a cluster. Each node runs a contiguous
slice of the task list and writes its own log file. Results from each node
must be aggregated manually after all jobs complete.

Typical CLI Usage
-----------------
    python run_libero_eval_task_comp.py \\
        --pretrained_checkpoint /path/to/checkpoint \\
        --comp_level l1 \\
        --num_trials_per_task 50 \\
        --seed 42 \\
        --run_id_note ablation_v1

Dependencies
------------
- libero        : LIBERO benchmark path registry and environment classes
- draccus       : CLI/YAML config parsing for Python dataclasses
- numpy         : Array math and state concatenation
- torch         : Deep learning backend + initial state loading
- tqdm          : Progress bars for episode and task loops
- wandb         : Weights & Biases experiment tracking (optional)

Author: Agostino Cardamone
"""


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import json       # JSON deserialisation (imported for completeness; not used directly in this file)
import logging    # Structured, levelled console and file logging
import os         # Filesystem operations: path joins, makedirs, listdir
import sys        # sys.path manipulation for local package discovery
import gc         # Explicit garbage collection after environment teardown

from collections import deque       # Fixed-capacity FIFO action chunk queue
from dataclasses import dataclass   # Declarative configuration struct via @dataclass
from enum import Enum               # Strongly-typed, string-comparable enum for task suites
from pathlib import Path            # Object-oriented cross-platform file paths
from typing import Optional, Union  # PEP 484 generic type annotations


# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

import draccus      # CLI wrapper that maps dataclass fields → argparse flags + YAML config
import numpy as np  # Numerical arrays: state concatenation, initial state casting
import torch        # Deep learning backend; also used to load .pruned_init state files
import tqdm         # Real-time progress bars wrapped around episode/task iterables


# =============================================================================
# LIBERO IMPORTS
# =============================================================================

from libero.libero import get_libero_path  # Returns absolute path to a named LIBERO
                                            # resource directory (e.g. "bddl_files",
                                            # "init_states")
from libero.libero.benchmark import Task   # NamedTuple definition for task metadata:
                                            # (name, language, problem, problem_folder,
                                            #  bddl_file, init_states_file)


# =============================================================================
# PROJECT ROOT RESOLUTION
# =============================================================================
# Make sibling top-level packages (experiments/, prismatic/) importable
# regardless of the working directory from which the script is invoked.

from pathlib import Path
current_file = Path(__file__).resolve()           # Absolute path of this script
project_root = current_file.parent.parent.parent  # Go up three directory levels to repo root
sys.path.insert(0, str(project_root))             # Prepend so local packages shadow any
                                                   # installed versions with the same name


# =============================================================================
# INTERNAL UTILITIES — LIBERO-SPECIFIC HELPERS
# =============================================================================

from experiments.libero.utils.libero_utils import (
    get_libero_dummy_action,    # Returns a neutral (zero) action for the warm-up phase
    get_libero_image,           # Extracts the third-person RGB frame from an observation dict
    get_libero_wrist_image,     # Extracts the wrist-mounted RGB frame from an observation dict
    quat2axisangle,             # Converts quaternion (qx,qy,qz,qw) → axis-angle (ax,ay,az)
    save_rollout_video,         # Renders a list of RGB frames to an MP4 replay video
    extract_command_from_bddl,  # Parses the `:language "..."` field from a BDDL file and
                                # returns the task instruction string
)


# =============================================================================
# INTERNAL UTILITIES — OPENVLA MODEL COMPONENTS
# =============================================================================

from experiments.openvla_utils import (
    get_action_head,            # Loads the L1 or DDIM-diffusion continuous action head
    get_noisy_action_projector, # Loads the noise-step conditioning projector for DDIM
    get_processor,              # Returns the VLA tokenizer + image pre-processor
    get_proprio_projector,      # Loads the MLP mapping 8-dim proprio → LLM embedding dim
    resize_image_for_policy,    # Resizes a raw NumPy image to the policy's expected (H, W)
)


# =============================================================================
# INTERNAL UTILITIES — GENERAL ROBOT EXPERIMENT HELPERS
# =============================================================================

from experiments.robot_utils import (
    DATE_TIME,                # ISO-formatted timestamp string captured at import time
    get_image_resize_size,    # Derives the (H, W) policy input size from config
    invert_gripper_action,    # Flips gripper sign to undo OpenVLA training convention
    normalize_gripper_action, # Maps gripper from [0,1] to [-1,+1] as expected by LIBERO env
    set_seed_everywhere,      # Sets Python, NumPy, and PyTorch random seeds globally
)


# =============================================================================
# VLA INFERENCE ENTRY POINTS
# =============================================================================

from experiments.openvla_utils import (
    get_vla,        # Instantiates the full VLA model from a checkpoint path
    get_vla_action, # Runs a forward pass and returns an action chunk
)

from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # Expected action chunk size (e.g. 8)


# =============================================================================
# LIBERO ENVIRONMENT CLASS
# =============================================================================

from libero.libero.envs import OffScreenRenderEnv  # MuJoCo/Robosuite environment that renders
                                                    # camera frames off-screen (no display needed)
                                                    # Used directly here instead of the benchmark
                                                    # factory, since tasks are defined by custom
                                                    # BDDL files outside the benchmark registry


# =============================================================================
# TASK COMPOSITION L1 — CUSTOM TASK REGISTRY
# =============================================================================

TASK_COMP_L1_TASKS = [
    {
        # COMPOSITION: ``plate → cabinet``
        # TRAINING PRIMITIVES:
        #   - push_plate_to_stove  → teaches plate manipulation
        #   - bowl/wine_bottle → cabinet  → teaches cabinet-placement skill
        # NOVEL ELEMENT: applying cabinet-placement to the plate object (new pair)
        "bddl_file": "put_the_plate_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # COMPOSITION: ``plate → stove``
        # TRAINING PRIMITIVES:
        #   - push_plate_to_stove  → teaches plate manipulation
        #   - bowl → stove  → teaches stove-placement skill
        # NOVEL ELEMENT: pick-and-place (not push) of the plate onto the stove
        "bddl_file": "put_the_plate_on_the_stove_task_comp_l1.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # COMPOSITION: ``cream_cheese → cabinet``
        # TRAINING PRIMITIVES:
        #   - cream_cheese → bowl  → teaches cream_cheese pick-place
        #   - bowl/wine_bottle → cabinet  → teaches cabinet-placement skill
        # NOVEL ELEMENT: placing cream_cheese directly on the cabinet (new pair)
        "bddl_file": "put_the_cream_cheese_on_top_of_the_cabinet_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # COMPOSITION: ``cream_cheese → plate``
        # TRAINING PRIMITIVES:
        #   - cream_cheese → bowl  → teaches cream_cheese pick-place
        #   - bowl → plate  → teaches plate as a placement target
        # NOVEL ELEMENT: cream_cheese placed on plate (new object–target pair)
        "bddl_file": "put_the_cream_cheese_on_the_plate_task_comp_l1.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # COMPOSITION: ``open top drawer + cream_cheese → inside``
        # TRAINING PRIMITIVES:
        #   - open drawer  → teaches drawer-opening skill
        #   - bowl → inside drawer  → teaches insert-into-drawer skill
        # NOVEL ELEMENT: inserting cream_cheese instead of bowl (swaps the object)
        "bddl_file": "open_the_top_drawer_and_put_the_cream_cheese_inside_task_comp_l1.bddl",
        "init_states_from": "open_the_top_drawer_and_put_the_bowl_inside",
    },
]
"""
list[dict]
    Registry of five L1 task composition scenarios.
    Each entry has:
    - ``"bddl_file"`` : filename of the custom BDDL file (resolved relative
      to the libero_goal BDDL directory at runtime).
    - ``"init_states_from"`` : stem of the ``.pruned_init`` file from which
      to load initial simulator states (must exist in the libero_goal
      init_states directory).
"""


# =============================================================================
# TASK COMPOSITION L2 — CUSTOM TASK REGISTRY
# =============================================================================

TASK_COMP_L2_TASKS = [
    {
        # L2 CHAIN: ``open middle drawer + put bowl inside``
        # Chain type: sequential (open actuator) + (pick-place into container)
        # Borrowing init states from the standard middle-drawer task for scene consistency
        "bddl_file": "open_the_middle_drawer_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "open_the_middle_drawer_of_the_cabinet",
    },
    {
        # L2 CHAIN: ``put bowl on stove + turn on stove``
        # Chain type: (pick-place) + (switch manipulation on the stove object)
        "bddl_file": "put_the_bowl_on_the_stove_task_comp_l2.bddl",
        "init_states_from": "put_the_bowl_on_the_stove",
    },
    {
        # L2 CHAIN: ``put cream_cheese in bowl + put bowl on plate``
        # Chain type: (pick-place A → B) + (pick-place B → C) — two sequential pick-places
        "bddl_file": "put_the_cream_cheese_in_the_bowl_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
    {
        # L2 CHAIN: ``push plate to stove front + put bowl on plate``
        # Chain type: (push/slide) + (pick-place) — heterogeneous primitive types chained
        "bddl_file": "push_the_plate_to_the_front_of_the_stove_task_comp_l2.bddl",
        "init_states_from": "push_the_plate_to_the_front_of_the_stove",
    },
    {
        # L2 CHAIN: ``put cream_cheese in bowl + put bowl on top of cabinet``
        # Chain type: (pick-place A → B) + (pick-place B → high target) — two pick-places
        # with an elevated final target requiring precise vertical motion
        "bddl_file": "put_the_bowl_on_top_of_the_cabinet_task_comp_l2.bddl",
        "init_states_from": "put_the_cream_cheese_in_the_bowl",
    },
]
"""
list[dict]
    Registry of five L2 task composition scenarios.
    Same schema as ``TASK_COMP_L1_TASKS``:
    - ``"bddl_file"`` : custom BDDL filename in the libero_goal directory.
    - ``"init_states_from"`` : stem of the borrowed ``.pruned_init`` file.
"""


# =============================================================================
# UNIFIED TASK REGISTRY
# =============================================================================

TASK_COMP_REGISTRY = {
    "l1": TASK_COMP_L1_TASKS,  # Object/target composition tasks
    "l2": TASK_COMP_L2_TASKS,  # Multi-step sequential chain tasks
}
"""
dict[str, list[dict]]
    Top-level registry mapping a composition level string (``"l1"`` or ``"l2"``)
    to its corresponding list of task definition dictionaries.
    Consumed by ``load_custom_tasks(comp_level)`` to select which task set to load.
"""


# =============================================================================
# GLOBAL STEP BUDGET
# =============================================================================

TASK_MAX_STEPS = 500
"""
int
    Hard maximum number of active simulator steps (excluding warm-up) allowed
    per episode across all task composition tasks. Set to 500 to accommodate
    the longer-horizon L2 chain tasks, which may require more steps than any
    individual libero_goal task (which caps at 300).
    If the environment signals ``done=True`` before this limit, the episode
    terminates early and is counted as a success.
"""


# =============================================================================
# MODULE-LEVEL LOGGER
# =============================================================================

# Configure root logger with ISO timestamp + severity + message, writing to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)  # Child logger namespaced to this module


# =============================================================================
# TASK SUITE ENUMERATION (retained for unnorm_key lookup compatibility)
# =============================================================================

class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    #################################################################################################################
    # LIBERO environment parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL       # Used for unnorm_key lookup
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    env_img_res: int = 256

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "/mnt/beegfs/a.cardamone7/outputs/logs"

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 42
    debug: bool = False
    comp_level: str = "l1"

    # Task subset (for splitting across nodes)
    task_start: int = 0
    task_end: int = -1  # -1 means all tasks
    # fmt: on


# =============================================================================
# DUAL-SINK LOG HELPER
# =============================================================================

def log_message(message: str, log_file=None) -> None:
    """
    Write a message to the module logger (stdout) and optionally to a log file.

    Ensures all status updates appear both in the console and in the persistent
    per-run ``.txt`` log. Uses ``flush()`` after each file write to guarantee
    data is persisted even if the process terminates unexpectedly.

    Parameters
    ----------
    message : str
        Text to log. A newline is appended automatically when writing to file.
    log_file : io.TextIOWrapper or None, optional
        Open writable file handle. When ``None``, the message goes only to
        the console logger.

    Returns
    -------
    None
    """
    logger.info(message)        # Always emit to console at INFO level
    if log_file:
        log_file.write(message + "\n")  # Newline-delimited for line-by-line readability
        log_file.flush()                # Force OS write-through; avoid buffering on crash


# =============================================================================
# MODEL INITIALISATION
# =============================================================================

def initialize_model(cfg: GenerateConfig):
    """
    Instantiate all neural network components required for policy inference.

    Loads each component conditionally based on architecture flags in ``cfg``,
    minimising GPU memory usage when optional modules are disabled. CUDA memory
    is cleared before loading to reduce fragmentation.

    Components loaded
    -----------------
    model : torch.nn.Module
        Full VLA backbone (visual encoder + LLM). Always loaded.
    proprio_projector : torch.nn.Module or None
        MLP mapping the 8-dim proprioceptive state into the LLM embedding
        space. Loaded when ``cfg.use_proprio=True``.
    action_head : torch.nn.Module or None
        Continuous action prediction head (L1 or DDIM). Loaded when
        ``cfg.use_l1_regression`` or ``cfg.use_diffusion`` is True. Falls back
        gracefully to ``None`` if the weight file is not found (e.g. the head
        is fused into a LoRA checkpoint).
    noisy_action_projector : torch.nn.Module or None
        Noise-step conditioning projector for DDIM. Loaded only when
        ``cfg.use_diffusion=True``.
    processor : transformers.ProcessorMixin or None
        VLA tokenizer + image pre-processor. Loaded when
        ``cfg.model_family == "openvla"``; after loading, ``check_unnorm_key``
        resolves and validates the action un-normalisation key.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration containing checkpoint path, architecture flags,
        and quantisation settings.

    Returns
    -------
    model : torch.nn.Module
        VLA backbone in evaluation mode.
    action_head : torch.nn.Module or None
    proprio_projector : torch.nn.Module or None
    noisy_action_projector : torch.nn.Module or None
    processor : transformers.ProcessorMixin or None
    """
    # Clear CUDA memory cache before loading large model weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    # Release all cached (but unused) CUDA memory
        torch.cuda.synchronize()    # Block until all pending CUDA kernels complete

    # ── 1. VLA backbone (always loaded) ──────────────────────────────────────
    model = get_vla(cfg)

    # ── 2. Proprioceptive projector (optional) ────────────────────────────────
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,  # Must align with the LLM's hidden dimension
            proprio_dim=8,  # 3 (EEF XYZ) + 3 (axis-angle) + 2 (gripper) = 8
        )

    # ── 3. Continuous action head (optional, graceful LoRA fallback) ──────────
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            logger.info("Action head loaded separately.")
        except (AssertionError, FileNotFoundError):
            # Head not found as a standalone file — assume it is fused into
            # the LoRA checkpoint weights and continue without error
            logger.warning("Action head not found as separate file; assuming integrated in model.")
            action_head = None

    # ── 4. DDIM noise projector (optional) ───────────────────────────────────
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # ── 5. Input processor (OpenVLA only) ─────────────────────────────────────
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)  # Resolve and validate action un-norm key in-place

    return model, action_head, proprio_projector, noisy_action_projector, processor


# =============================================================================
# ACTION UN-NORMALISATION KEY RESOLUTION
# =============================================================================

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """
    Resolve and validate the action un-normalisation key in the VLA model.

    The model's ``norm_stats`` dict stores per-dimension action statistics
    (mean, std) under dataset-specific string keys. At inference time, raw
    predicted actions must be de-standardised using those statistics. This
    function identifies the correct key and writes it back into ``cfg.unnorm_key``.

    Resolution Priority
    -------------------
    1. If ``cfg.unnorm_key`` is set and exists in ``model.norm_stats`` → use it.
    2. Otherwise derive from ``cfg.task_suite_name`` (default: ``"libero_goal"``).
    3. Try ``"{key}_no_noops"`` if the plain key is absent.
    4. Try ``"{key}_noops"`` as a second fallback variant.
       (This script adds this third fallback compared to the standard eval.)
    5. If no key is found → raise ``AssertionError``.

    Side Effects
    ------------
    Mutates ``cfg.unnorm_key`` in place with the resolved key string.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration object. ``cfg.unnorm_key`` is updated.
    model : torch.nn.Module
        Loaded VLA model exposing a ``norm_stats`` dict attribute.

    Raises
    ------
    AssertionError
        If no valid key can be found in ``model.norm_stats``.
    """
    # Log available keys and current config value for debugging
    logger.info(f"Available norm_stats keys: {list(model.norm_stats.keys())}")
    logger.info(f"cfg.unnorm_key from config: '{cfg.unnorm_key}'")

    # Priority 1: User-specified key that already exists
    if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
        logger.info(f"Using user-specified unnorm_key: {cfg.unnorm_key}")
        return

    # Priority 2: Derive from task suite name
    unnorm_key = cfg.task_suite_name   # e.g. "libero_goal"

    # Priority 3: Try the "_no_noops" dataset variant
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    # Priority 4: Try the "_noops" variant (unique to this script vs. standard eval)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_noops"

    # Fail with a clear message if no valid key was found
    assert unnorm_key in model.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in VLA `norm_stats`!"
    )

    cfg.unnorm_key = unnorm_key  # Persist resolved key back into the config


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(cfg: GenerateConfig):
    """
    Initialise per-run logging to a local ``.txt`` file and optionally to W&B.

    Constructs a unique, human-readable run ID embedding the composition level,
    model family, timestamp, and an optional user-provided note. This ID is
    used as both the log filename and the W&B run name.

    Run ID Format
    -------------
    ``EVAL-task_comp_{comp_level}-{model_family}-{DATE_TIME}[--{run_id_note}]``

    Example: ``EVAL-task_comp_l1-openvla-20260416_112800--ablation_v1``

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration providing composition level, model family, optional
        run note, and logging destinations.

    Returns
    -------
    log_file : io.TextIOWrapper
        Open file handle (write mode) for the ``.txt`` log file.
        Caller must close it when the evaluation completes.
    local_log_filepath : str
        Absolute path of the created ``.txt`` log file.
    run_id : str
        Unique run identifier used for the log filename and W&B run name.
    """
    # Build unique run ID from composition level + model family + timestamp
    run_id = f"EVAL-task_comp_{cfg.comp_level}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"   # Append user annotation if provided

    # Create log directory (including any missing parents) and open the log file
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")  # Open in write mode; truncates existing file
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Optionally initialise a W&B run for real-time metric streaming
    if cfg.use_wandb:
        import wandb   # Imported lazily; wandb is optional
        wandb.init(
            entity=cfg.wandb_entity,   # W&B user/team
            project=cfg.wandb_project, # W&B project board
            name=run_id,               # Run name in the W&B UI
        )

    return log_file, local_log_filepath, run_id


# =============================================================================
# OBSERVATION PREPARATION
# =============================================================================

def prepare_observation(obs: dict, resize_size: tuple) -> tuple:
    """
    Convert a raw LIBERO observation dictionary into the format expected by
    the VLA model for a single inference call.

    The VLA requires:
    - Two resized RGB camera frames (third-person view + wrist view).
    - An 8-dimensional proprioceptive state vector.

    Proprioceptive State Layout
    ---------------------------
    The state vector is constructed by concatenating:
    - ``obs["robot0_eef_pos"]``        : (3,) end-effector XYZ position
    - ``quat2axisangle(obs["robot0_eef_quat"])``: (3,) axis-angle rotation
    - ``obs["robot0_gripper_qpos"]``   : (2,) gripper joint positions
    Total: 8 dimensions.

    Parameters
    ----------
    obs : dict
        Raw observation dict from ``env.step()`` or ``env.reset()``.
    resize_size : tuple of (int, int)
        Target (H, W) resolution for policy input images.

    Returns
    -------
    observation : dict
        Keys: ``"full_image"`` (H',W',3), ``"wrist_image"`` (H',W',3),
        ``"state"`` (8,).
    img : np.ndarray
        Original (un-resized) third-person frame for video replay recording.
    """
    # ── Extract raw frames from observation ───────────────────────────────────
    img = get_libero_image(obs)             # Third-person camera RGB (H, W, 3)
    wrist_img = get_libero_wrist_image(obs) # Wrist camera RGB (H, W, 3)

    # ── Resize to policy's expected input resolution ───────────────────────────
    img_resized       = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # ── Assemble the observation dict consumed by get_vla_action ─────────────
    observation = {
        "full_image":  img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate((
            obs["robot0_eef_pos"],                         # (3,) XYZ position
            quat2axisangle(obs["robot0_eef_quat"]),        # (3,) axis-angle rotation
            obs["robot0_gripper_qpos"],                    # (2,) gripper joints
        )),  # → shape (8,)
    }

    # Return processed observation AND original image (for replay video)
    return observation, img


# =============================================================================
# ACTION POST-PROCESSING
# =============================================================================

def process_action(action: np.ndarray, model_family: str) -> np.ndarray:
    """
    Post-process a raw predicted action before sending it to the simulator.

    Two sequential sign/scale transformations reconcile the VLA training
    convention with the LIBERO environment's expected action format:

    Step 1 — Gripper normalisation
        The action head outputs the gripper dimension in [0, 1]. LIBERO's
        ``env.step()`` expects [-1, +1]. ``normalize_gripper_action`` with
        ``binarize=True`` performs this remapping and snaps to exactly ±1.

    Step 2 — OpenVLA sign inversion
        During OpenVLA training the gripper dimension sign was flipped to
        unify conventions across datasets (0 = close, 1 = open). After Step 1
        the LIBERO convention is already correct, but since the model learned
        with the flipped convention, ``invert_gripper_action`` restores the
        original sign so the simulator receives the correct command.

    Parameters
    ----------
    action : np.ndarray of shape (action_dim,)
        Raw predicted action vector from the model.
    model_family : str
        Model family identifier. Sign inversion only applied for ``"openvla"``.

    Returns
    -------
    action : np.ndarray of shape (action_dim,)
        Post-processed action ready for ``env.step()``.
    """
    action = normalize_gripper_action(action, binarize=True)  # [0,1] → {-1, +1}
    if model_family == "openvla":
        action = invert_gripper_action(action)                 # Undo training sign flip
    return action

# =============================================================================
# CUSTOM TASK LOADER
# =============================================================================

def load_custom_tasks(comp_level: str = "l1") -> list:
    """
    Build ``Task`` NamedTuples and load pre-recorded initial states for each
    task composition scenario at the specified level.

    This function replaces the standard ``benchmark.get_benchmark_dict()``
    registry lookup used in ``run_libero_eval.py``. Instead of relying on
    the LIBERO benchmark to discover tasks, it:

    1. Reads task definitions from ``TASK_COMP_REGISTRY[comp_level]``.
    2. Resolves each BDDL file path relative to the libero_goal BDDL directory.
    3. Extracts the task language instruction by parsing the ``:language "..."``
       field from the custom BDDL file via ``extract_command_from_bddl``.
    4. Constructs a ``Task`` NamedTuple with the custom BDDL filename and the
       ``.pruned_init`` filename of the borrowed reference task.
    5. Loads the initial states tensor from the corresponding ``.pruned_init``
       file in the libero_goal init_states directory using ``torch.load``.

    The ``.pruned_init`` files are pre-recorded expert-demonstration initial
    states from the standard libero_goal training tasks. They are reused here
    because all custom tasks share the same libero_goal scene layout, meaning
    the object placements and robot configurations from those tasks are
    valid starting points for the custom composition tasks.

    Parameters
    ----------
    comp_level : str, default "l1"
        Composition level key: ``"l1"`` for object/target composition tasks,
        ``"l2"`` for multi-step chain tasks.

    Returns
    -------
    custom_tasks : list of dict
        Each dict contains:
        - ``"task"`` : ``Task`` NamedTuple with fields:
            - ``name``            : BDDL filename without ``.bddl`` extension
            - ``language``        : task instruction string from the BDDL file
            - ``problem``         : ``"Libero"``
            - ``problem_folder``  : ``"libero_goal"``
            - ``bddl_file``       : BDDL filename (e.g. ``"put_the_plate_on_top_of_the_cabinet_task_comp_l1.bddl"``)
            - ``init_states_file``: ``.pruned_init`` filename used for initial states
        - ``"init_states"`` : ``torch.Tensor`` or list of states loaded from
            the ``.pruned_init`` file. Index ``i`` gives the initial state for
            episode ``i``.
        - ``"task_description"`` : task instruction string (same as ``task.language``)
        - ``"bddl_path"``        : absolute path to the custom BDDL file

    Raises
    ------
    AssertionError
        If the BDDL file or ``.pruned_init`` file for any task does not exist
        on disk. Fails fast with a descriptive path for easy debugging.
    AssertionError
        If ``extract_command_from_bddl`` cannot find a ``:language`` field in
        the BDDL file.
    """
    # Resolve absolute paths to BDDL and init_states directories for libero_goal
    bddl_dir  = os.path.join(get_libero_path("bddl_files"), "libero_goal")
    init_dir  = os.path.join(get_libero_path("init_states"), "libero_goal")

    custom_tasks = []

    for task_def in TASK_COMP_REGISTRY[comp_level]:
        bddl_filename = task_def["bddl_file"]      # Custom BDDL filename
        init_from     = task_def["init_states_from"]  # Reference task stem for init states

        # Build absolute path to the custom BDDL file
        bddl_path = os.path.join(bddl_dir, bddl_filename)
        assert os.path.exists(bddl_path), f"BDDL file not found: {bddl_path}"

        # Parse the `:language "..."` instruction string from the BDDL file
        task_description = extract_command_from_bddl(bddl_path)
        assert task_description is not None, (
            f"Could not extract :language field from {bddl_path}"
        )

        # Build Task NamedTuple: name = BDDL stem (without .bddl extension)
        task_name = bddl_filename.replace(".bddl", "")
        task = Task(
            name=task_name,
            language=task_description,
            problem="Libero",
            problem_folder="libero_goal",
            bddl_file=bddl_filename,
            init_states_file=f"{init_from}.pruned_init",  # Maps to borrowed init states
        )

        # Load initial states from the reference task's .pruned_init file
        init_states_path = os.path.join(init_dir, f"{init_from}.pruned_init")
        assert os.path.exists(init_states_path), (
            f"Init states file not found: {init_states_path}"
        )
        # weights_only=False is required because .pruned_init files may contain
        # non-tensor objects (e.g. numpy arrays, dicts) that torch.load must
        # unpickle with full Python functionality
        init_states = torch.load(init_states_path, weights_only=False)

        custom_tasks.append({
            "task":             task,
            "init_states":      init_states,
            "task_description": task_description,
            "bddl_path":        bddl_path,
        })

    return custom_tasks


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def create_env_from_bddl(bddl_path: str, resolution: int = 256):
    """
    Instantiate a LIBERO ``OffScreenRenderEnv`` directly from a BDDL file path.

    Unlike the standard evaluation script (which uses the LIBERO benchmark
    factory ``get_libero_env``), this function constructs the environment
    directly by passing the BDDL file path to ``OffScreenRenderEnv``. This is
    necessary because the custom composition tasks are not registered in the
    LIBERO benchmark dictionary and therefore cannot be looked up by name.

    The environment is seeded with a fixed value of ``0`` at creation time.
    The actual initial simulator state is set per-episode via
    ``env.set_init_state(initial_state)`` in the episode runner, so this seed
    only affects environment initialisation (asset loading, RNG state), not
    the object placement at episode start.

    Parameters
    ----------
    bddl_path : str
        Absolute path to the custom BDDL file defining the task's goal
        conditions, object set, and scene layout.
    resolution : int, default 256
        Pixel resolution (H = W) for rendered camera frames. Applied to both
        ``camera_heights`` and ``camera_widths``.

    Returns
    -------
    env : libero.libero.envs.OffScreenRenderEnv
        Instantiated MuJoCo simulation environment, ready to be reset.
    """
    env_args = {
        "bddl_file_name": bddl_path,    # Absolute path to the custom BDDL file
        "camera_heights": resolution,   # Vertical render resolution in pixels
        "camera_widths":  resolution,   # Horizontal render resolution in pixels
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)   # Fixed seed for deterministic asset loading; actual state set later
    return env


# =============================================================================
# SINGLE EPISODE RUNNER
# =============================================================================

def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size: tuple,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
) -> tuple:
    """
    Execute a single evaluation episode in a LIBERO simulation environment.

    This is the innermost execution unit of the evaluation pipeline. It runs
    exactly one rollout from an initial state to either task completion or
    step-budget exhaustion, collecting frames and end-effector positions for
    downstream video saving and trajectory analysis.

    Architecture: Open-Loop Action Chunking
    ----------------------------------------
    The policy is queried infrequently — once per ``cfg.num_open_loop_steps``
    simulator steps — rather than at every step. Each query returns a batch
    of ``NUM_ACTIONS_CHUNK`` predicted actions. These are stored in a FIFO
    ``deque`` and consumed one-by-one until the queue empties, at which point
    the policy is re-queried. This reduces VLA forward-pass frequency and
    improves throughput significantly on GPU-limited hardware.

    The action queue has ``maxlen=cfg.num_open_loop_steps`` so it cannot grow
    unboundedly; if somehow more actions are added than the capacity allows,
    the oldest entries are automatically discarded.

    Episode Phases
    --------------
    **Phase 1 — Environment Reset (steps 0 to num_steps_wait-1)**
        Warm-up dummy actions are executed to let physics settle.
        ``get_libero_dummy_action`` returns a zero-velocity, neutral gripper
        command. No observations are collected during this phase; the loop
        simply advances the simulator clock. This prevents the policy from
        seeing observations taken while objects are still bouncing from the
        reset.

    **Phase 2 — Active Control (steps num_steps_wait to max_steps + num_steps_wait)**
        Frames and states are recorded at every step. Policy queries happen
        whenever the action queue is empty (i.e. at the start of each chunk).
        The first few policy queries (t < 50) produce verbose action logging
        to aid debugging. If the environment returns ``done=True``, the episode
        immediately terminates and is marked as a success.

    **Exception Handling**
        The entire active-control phase is wrapped in a broad ``try/except``.
        Any uncaught exception (CUDA OOM, MuJoCo crash, tensor shape mismatch,
        etc.) is logged with ``log_message`` and the episode is returned as
        failed (``success=False``). This prevents a single corrupted episode
        from aborting the entire evaluation run.

    Differences from ``run_libero_eval.run_episode``
    -------------------------------------------------
    - ``max_steps = TASK_MAX_STEPS`` (module-level constant 500) rather than
      a per-task lookup from a ``{task_name: max_steps}`` dict. All task
      composition tasks share the same step budget.
    - The ``num_open_loop_steps != NUM_ACTIONS_CHUNK`` warning is printed here
      (not in the main function) since this script has no per-level main loop.

    Parameters
    ----------
    cfg : GenerateConfig
        Full evaluation configuration. Key fields used:
        - ``cfg.num_steps_wait``      : warm-up steps before active control
        - ``cfg.num_open_loop_steps`` : actions per policy query
        - ``cfg.model_family``        : gripper post-processing convention
        - ``cfg.use_film``            : pass FiLM flag to action inference
    env : libero.libero.envs.OffScreenRenderEnv
        Instantiated MuJoCo simulation environment. Must already be created;
        this function calls ``env.reset()`` internally.
    task_description : str
        Natural language instruction conditioning the VLA forward pass
        (e.g. ``"put the cream cheese on top of the cabinet"``).
    model : torch.nn.Module
        Loaded VLA backbone in evaluation mode (``model.eval()``).
    resize_size : tuple of (int, int)
        Target (H, W) pixel resolution for policy input images; passed
        directly to ``prepare_observation``.
    processor : transformers.ProcessorMixin or None
        VLA tokenizer and image pre-processor. ``None`` if not applicable.
    action_head : torch.nn.Module or None
        Continuous action head (L1 regression or DDIM); ``None`` if the head
        is fused into the model or not used.
    proprio_projector : torch.nn.Module or None
        Proprioceptive state MLP; ``None`` if ``cfg.use_proprio=False``.
    noisy_action_projector : torch.nn.Module or None
        DDIM noise-step conditioning projector; ``None`` if not using diffusion.
    initial_state : np.ndarray or None
        Pre-recorded simulator state loaded from a ``.pruned_init`` file.
        When provided, ``env.set_init_state(initial_state)`` is called after
        ``env.reset()`` to place the robot and objects in the recorded
        configuration. When ``None``, the default post-reset configuration
        is used (stochastic, hardware-dependent).
    log_file : io.TextIOWrapper or None
        Open writable log file. Episode-level errors are written here.

    Returns
    -------
    success : bool
        ``True`` if and only if ``env.step()`` returned ``done=True`` before
        the step budget (``TASK_MAX_STEPS + cfg.num_steps_wait``) was reached.
        ``False`` on timeout or any uncaught exception.
    replay_images : list of np.ndarray
        Ordered list of third-person RGB frames (H, W, 3), one per active
        step. Empty if an exception occurs before any observation is collected.
        Used by ``save_rollout_video`` to render the episode MP4.
    replay_states : list of np.ndarray
        Ordered list of end-effector XYZ positions (3,), one per active step.
        Parallel to ``replay_images``; used for trajectory visualisation.
    """

    # ── Phase 1: Reset simulator to a clean state ─────────────────────────────
    env.reset()  # Deallocate previous episode state; randomise physics noise if any

    if initial_state is not None:
        # Deterministically restore a pre-recorded simulator state.
        # set_init_state positions both robot joints and scene objects.
        obs = env.set_init_state(initial_state)
    else:
        # No specific state provided: use the default post-reset observation.
        # In practice this should not occur, since all tasks supply .pruned_init states.
        obs = env.get_observation()

    # Warn if the step-execution configuration mismatches the training constant.
    # A mismatch does not crash evaluation but can degrade success rates because
    # the policy was trained under a specific chunk execution frequency.
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) != "
            f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})"
        )

    # Fixed-capacity FIFO queue; maxlen prevents queue growth beyond one chunk
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # ── Initialise episode tracking variables ────────────────────────────────
    t             = 0        # Global step counter (includes warm-up steps)
    replay_images = []       # Collects third-person frames for MP4 video
    replay_states = []       # Collects EEF XYZ positions for trajectory logging
    max_steps     = TASK_MAX_STEPS  # 500 — shared budget for all composition tasks

    success = False  # Default: episode failed (timeout or exception)

    # ── Phase 2: Main episode loop ────────────────────────────────────────────
    try:
        # Loop runs for warm-up phase + active control phase
        while t < max_steps + cfg.num_steps_wait:

            # ── Warm-up: dummy actions, no data collection ────────────────────
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(
                    get_libero_dummy_action(cfg.model_family)
                )
                t += 1
                continue  # Skip the rest; do not record observations during warm-up

            # ── Active phase: prepare and record observation ──────────────────
            # prepare_observation resizes images and assembles the 8-dim proprio vector
            observation, img = prepare_observation(obs, resize_size)

            replay_images.append(img)                     # Third-person frame for video
            replay_states.append(obs["robot0_eef_pos"])   # EEF position (3,) for trajectory

            # ── Policy query: re-fill action queue when empty ─────────────────
            if len(action_queue) == 0:
                # get_vla_action runs a full VLA forward pass and returns a list of
                # NUM_ACTIONS_CHUNK action vectors, each of shape (action_dim,)
                actions = get_vla_action(
                    cfg,
                    model,
                    processor,
                    observation,
                    task_description,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )

                # Verbose action logging for the first 50 active steps only,
                # to aid debugging without flooding the log with thousands of lines
                if t < 50:
                    logger.info(f"Step {t}: action[0] = {actions[0][:4]}...")

                # Extend the queue with the full predicted chunk
                action_queue.extend(actions)

            # ── Execute next action from queue ───────────────────────────────
            action = action_queue.popleft()             # FIFO: consume oldest action
            action = process_action(action, cfg.model_family)  # Normalise + invert gripper

            # env.step expects a Python list (not a NumPy array) in LIBERO's interface
            obs, reward, done, info = env.step(action.tolist())

            if done:
                # Environment signals that all goal conditions in the BDDL are satisfied
                success = True
                break   # Exit immediately; no need to exhaust the budget

            t += 1  # Advance only on active (non-warm-up) steps counted within the budget

    except Exception as e:
        # Broad catch: GPU OOM, MuJoCo crash, tensor shape mismatch, etc.
        # The episode is marked as failed but evaluation continues with the next episode.
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images, replay_states


# =============================================================================
# SINGLE TASK RUNNER
# =============================================================================

def run_custom_task(
    cfg: GenerateConfig,
    task_info: dict,
    task_idx: int,
    num_tasks: int,
    model,
    resize_size: tuple,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes: int = 0,
    total_successes: int = 0,
    log_file=None,
) -> tuple:
    """
    Run the full evaluation protocol for a single custom task composition task.

    This function manages the complete lifecycle of evaluating one task:
    environment creation, the episode loop across all ``cfg.num_trials_per_task``
    trials, per-episode video saving, real-time logging, optional W&B metric
    streaming, environment teardown, and garbage collection.

    Compared to ``run_task`` in ``run_libero_eval.py``, this function is
    simpler because:
    - There is exactly one environment configuration per task (no command
      variants or version loops).
    - The environment is created directly from a custom BDDL file rather than
      through the LIBERO benchmark factory.
    - Initial states are loaded unconditionally from the task's ``.pruned_init``
      file (no ``cfg.initial_states_path`` branching logic).

    Episode Indexing and State Access
    ----------------------------------
    Initial states are indexed directly from ``task_info["init_states"]`` using
    ``episode_idx`` as an integer index. The ``.pruned_init`` format stores
    states as a tensor or list where ``init_states[i]`` is the complete
    MuJoCo simulator state for trial ``i``. All 50 (or ``num_trials_per_task``)
    trials are run in order from index 0 to N-1.

    Counter Semantics
    -----------------
    ``total_episodes`` and ``total_successes`` are **global accumulators**
    passed in from the outer task loop in ``eval_task_comp``. They are updated
    in place on every episode and returned to the caller so they can be threaded
    through successive ``run_custom_task`` calls. This produces a running
    cross-task success rate visible in the console in real time.

    ``task_episodes`` and ``task_successes`` are local counters that track
    performance on this specific task only, used to compute the per-task
    success rate returned to the caller.

    Video Save Path
    ---------------
    Rollout videos are saved to a structured directory:

        ``/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/task_composition/
          openvla-oft/task_comp_{comp_level}/run_{run_id_note}/``

    This path is hardcoded via ``cfg.comp_level`` and ``cfg.run_id_note``.
    The ``total_episodes`` integer provides a unique, sortable filename
    identifier for each video. Successful episodes are named differently
    from failed ones by ``save_rollout_video``.

    Environment Teardown
    --------------------
    ``env.close()`` is called inside a ``try/except`` after all episodes
    complete. Some LIBERO environment versions do not implement ``close()``,
    so the exception is caught and logged as a non-fatal warning. ``gc.collect()``
    is called unconditionally afterward to reclaim any MuJoCo C-level heap
    allocations before the next task's environment is instantiated.

    W&B Metric Logging
    ------------------
    If ``cfg.use_wandb=True``, per-task success rate and episode count are
    streamed to W&B using the task description string as the metric key prefix.
    This produces one group of W&B metrics per evaluated task, enabling
    per-task success curves in the W&B dashboard.

    Parameters
    ----------
    cfg : GenerateConfig
        Full evaluation configuration.
    task_info : dict
        Task definition dict returned by ``load_custom_tasks``. Expected keys:
        - ``"task"``             : ``Task`` NamedTuple with metadata
        - ``"init_states"``      : indexable sequence of pre-recorded states
        - ``"task_description"`` : natural language instruction string
        - ``"bddl_path"``        : absolute path to the custom BDDL file
    task_idx : int
        0-indexed position of this task in the evaluated task list. Used only
        for display in the log header (``TASK {task_idx+1}/{num_tasks}``).
    num_tasks : int
        Total number of tasks being evaluated in this run. Used only for display.
    model : torch.nn.Module
        Loaded VLA backbone in evaluation mode.
    resize_size : tuple of (int, int)
        Target (H, W) resolution for policy input images.
    processor : transformers.ProcessorMixin or None
        VLA tokenizer/image processor.
    action_head : torch.nn.Module or None
        Continuous action head.
    proprio_projector : torch.nn.Module or None
        Proprioceptive state projector.
    noisy_action_projector : torch.nn.Module or None
        DDIM diffusion noise projector.
    total_episodes : int, default 0
        Running total of episodes completed across all tasks so far.
        Updated each episode and returned to the caller.
    total_successes : int, default 0
        Running total of successful episodes across all tasks so far.
        Updated each episode and returned to the caller.
    log_file : io.TextIOWrapper or None
        Open writable log file for structured per-episode status messages.

    Returns
    -------
    total_episodes : int
        Updated grand total of all episodes run (including this task).
    total_successes : int
        Updated grand total of all successes (including this task).
    task_description : str
        The natural language instruction for this task. Used by ``eval_task_comp``
        as the dictionary key in ``task_results``.
    task_success_rate : float
        Per-task success rate in [0.0, 1.0]. Zero-safe: returns 0.0 when
        no episodes were run.
    task_episodes : int
        Number of episodes executed for this task. Always equals
        ``cfg.num_trials_per_task`` unless an edge case prevents execution.
    """

    # ── Unpack task info dict ─────────────────────────────────────────────────
    task             = task_info["task"]
    init_states      = task_info["init_states"]
    task_description = task_info["task_description"]
    bddl_path        = task_info["bddl_path"]

    # ── Instantiate environment from the custom BDDL file ────────────────────
    # A single environment instance is shared across all episodes for this task.
    # Each episode resets via env.reset() + env.set_init_state() internally.
    env = create_env_from_bddl(bddl_path, resolution=cfg.env_img_res)

    # ── Log task header ───────────────────────────────────────────────────────
    log_message("=" * 80, log_file)
    log_message(f"TASK {task_idx + 1}/{num_tasks} (Task Composition {cfg.comp_level.upper()})", log_file)
    log_message(f"BDDL: {task.bddl_file}", log_file)
    log_message(f"Command: {task_description}", log_file)
    log_message(f"Init states from: {task.init_states_file}", log_file)
    log_message("=" * 80, log_file)

    # ── Per-task counters (local to this task) ────────────────────────────────
    task_episodes  = 0
    task_successes = 0

    # =========================================================================
    # EPISODE LOOP
    # =========================================================================

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):

        # Retrieve the pre-recorded initial state for this trial index.
        # init_states is a tensor/list from .pruned_init; integer indexing returns
        # a single state vector (MuJoCo qpos + qvel + object poses).
        initial_state = init_states[episode_idx]

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # ── Execute episode ───────────────────────────────────────────────────
        success, replay_images, replay_states = run_episode(
            cfg, env, task_description, model, resize_size,
            processor, action_head, proprio_projector,
            noisy_action_projector, initial_state, log_file,
        )

        # ── Update counters ───────────────────────────────────────────────────
        task_episodes  += 1
        total_episodes += 1
        if success:
            task_successes  += 1
            total_successes += 1

        # ── Save rollout video ────────────────────────────────────────────────
        # The video directory encodes both the composition level and the run note
        # so that results from different ablation runs are stored separately.
        # total_episodes provides a globally unique, monotonically increasing
        # numeric index for video filenames, ensuring no overwrites across tasks.
        save_rollout_video(
            {'image': replay_images, 'states': replay_states},
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
            dataset_name="task_comp_l1",
            run=cfg.run_id_note,
            custom_video_dir=(
                f"/mnt/beegfs/a.cardamone7/outputs/rollouts/libero_goal/"
                f"task_composition/openvla-oft/task_comp_{cfg.comp_level}/"
                f"run_{cfg.run_id_note}"
            ),
        )

        # ── Log per-episode result ────────────────────────────────────────────
        # Running cross-task rate is logged after every episode for real-time monitoring
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed: {total_episodes}", log_file)
        log_message(
            f"# successes: {total_successes} "
            f"({total_successes / total_episodes * 100:.1f}%)",
            log_file,
        )

    # =========================================================================
    # POST-TASK METRICS
    # =========================================================================

    # Per-task success rate: zero-safe guard prevents ZeroDivisionError if all
    # episodes were skipped (should not occur, but defensive)
    task_success_rate  = float(task_successes)  / float(task_episodes)  if task_episodes  > 0 else 0.0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    log_message(
        f"Task success rate: {task_success_rate:.4f} ({task_success_rate * 100:.1f}%)",
        log_file,
    )
    log_message(f"Running total success rate: {total_success_rate:.4f}", log_file)

    # ── Optional W&B metric push ──────────────────────────────────────────────
    if cfg.use_wandb:
        import wandb  # Lazy import: wandb is optional and heavy
        wandb.log({
            # Task description used as metric key prefix for per-task dashboards
            f"success_rate/{task_description}": task_success_rate,
            f"num_episodes/{task_description}": task_episodes,
        })

    # =========================================================================
    # ENVIRONMENT TEARDOWN
    # =========================================================================

    try:
        env.close()
        log_message("Environment closed successfully", log_file)
    except Exception as e:
        # Non-fatal: some LIBERO env versions don't implement close()
        # (AttributeError) or raise on an already-closed environment.
        log_message(f"Warning: Error closing environment: {e}", log_file)

    # Force CPython GC to reclaim MuJoCo C-heap memory before the next
    # environment is created; without this, fragmentation can cause OOM
    # errors on long evaluation runs with many sequential environments.
    gc.collect()

    # Return updated global counters + per-task summary metrics
    return total_episodes, total_successes, task_description, task_success_rate, task_episodes


# =============================================================================
# RESULTS TABLE PRINTER
# =============================================================================

def print_results_table(task_results: dict, all_results: dict) -> None:
    """
    Print a formatted ASCII results table for a task composition evaluation run.

    Renders a two-section summary to ``sys.stdout``:

    1. **Per-task table**: one row per evaluated task showing task description,
       per-task success rate as a percentage, absolute counts (successes/trials),
       and episode count.
    2. **Overall row**: aggregate success rate and totals across all tasks.

    Output Format
    -------------
    .. code-block:: text

        ====================================================================================================
        TASK COMPOSITION L1 - RESULTS TABLE
        ====================================================================================================
        Task                                                         |      Success Rate    | Episodes
        ----------------------------------------------------------------------------------------------------
        put the plate on top of the cabinet                          |       84.0% (42/50)  |       50
        put the plate on the stove                                   |       76.0% (38/50)  |       50
        put the cream cheese on top of the cabinet                   |       68.0% (34/50)  |       50
        put the cream cheese on the plate                            |       72.0% (36/50)  |       50
        open the top drawer and put the cream cheese inside          |       60.0% (30/50)  |       50
        ----------------------------------------------------------------------------------------------------
        OVERALL                                                      |       72.0% (180/250)|      250
        ====================================================================================================

    Column Layout
    -------------
    - ``Task`` : left-aligned in a 60-character field. Long descriptions are
      truncated by the Python format string if they exceed 60 characters.
    - ``Success Rate`` : right-aligned in a 20-character field, formatted as
      ``{rate:.1%} ({successes}/{trials})``. Successes are back-computed from
      ``result['success_rate'] * result['episodes']`` and cast to ``int``.
    - ``Episodes`` : right-aligned in an 8-character field. Redundant with the
      count inside Success Rate but kept for quick column scanning.

    Parameters
    ----------
    task_results : dict
        Nested dictionary mapping task description string to per-task results:
        ``{task_description: {'success_rate': float, 'episodes': int}}``.
        Iterated in insertion order (Python 3.7+), so tasks appear in the
        order they were evaluated.
    all_results : dict
        Aggregate results across all tasks:
        ``{'success_rate': float, 'total_episodes': int, 'total_successes': int}``.
        Pre-computed in ``eval_task_comp`` and passed directly to avoid
        recomputing from ``task_results``.

    Returns
    -------
    None
        All output is written to ``sys.stdout`` via ``print()``.
    """
    print("\n" + "=" * 100)
    print("TASK COMPOSITION L1 - RESULTS TABLE")
    print("=" * 100)

    # Column header
    print(f"{'Task':<60} | {'Success Rate':>20} | {'Episodes':>8}")
    print("-" * 100)

    # ── Per-task rows ─────────────────────────────────────────────────────────
    for task_name, result in task_results.items():
        sr       = result['success_rate']         # float in [0.0, 1.0]
        eps      = result['episodes']             # int, e.g. 50
        successes = int(sr * eps)                 # Recover int count from rate × episodes

        # Format: "84.0% (42/50)" inside the 20-char Success Rate column
        print(f"{task_name:<60} | {sr:>11.1%} ({successes:>2}/{eps:<2}) | {eps:>8}")

    print("-" * 100)

    # ── Overall / aggregate row ───────────────────────────────────────────────
    overall_sr   = all_results['success_rate']
    overall_succ = all_results['total_successes']
    overall_eps  = all_results['total_episodes']

    # :>3 and :<3 right/left-aligns counts within their width; adjust if
    # total episodes exceeds 999 (unlikely with standard 5-task × 50-trial runs)
    print(
        f"{'OVERALL':<60} | {overall_sr:>11.1%} ({overall_succ:>3}/{overall_eps:<3}) | {overall_eps:>8}"
    )
    print("=" * 100)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

@draccus.wrap()
def eval_task_comp(cfg: GenerateConfig) -> float:
    """
    Main entry point for evaluating a trained VLA policy on task composition
    generalization benchmarks (L1 or L2).

    Decorated with ``@draccus.wrap()``, which wraps the function so that
    ``GenerateConfig`` fields are automatically populated from CLI flags
    (``--field_name value``) and/or a YAML config file (``--config path.yaml``)
    when the script is executed. The function can also be called programmatically
    by passing a ``GenerateConfig`` instance directly.

    High-Level Orchestration
    -------------------------
    1. **Debugger hook** (optional): pauses startup and listens for a
       debugpy remote debugger on port 5678 when ``cfg.debug=True``. Useful
       for stepping through episode execution on a cluster node.
    2. **Validation**: asserts that ``pretrained_checkpoint`` is provided and
       that 8-bit and 4-bit quantisation are not enabled simultaneously.
    3. **Seed**: ``set_seed_everywhere`` makes Python, NumPy, and PyTorch
       fully deterministic before any model or environment initialisation.
    4. **Model loading**: ``initialize_model`` loads the VLA backbone and all
       optional components (action head, proprio projector, noise projector,
       processor) and resolves the action un-normalisation key.
    5. **Resize size**: ``get_image_resize_size`` derives the (H, W) pair
       the policy expects for its input images from the model config.
    6. **Task loading**: ``load_custom_tasks`` builds ``Task`` NamedTuples
       and loads pre-recorded initial states for all tasks at
       ``cfg.comp_level`` (L1 or L2).
    7. **Task subset selection**: ``all_custom_tasks[task_start:task_end]``
       selects the slice of tasks this node is responsible for. The full task
       list is the same on all nodes; each node runs a different contiguous
       slice and writes its own log. Results must be aggregated manually.
    8. **Pre-logging** (to console only, before the log file is opened): prints
       the total number of tasks, the active slice, and a numbered list of task
       descriptions with their BDDL filenames.
    9. **Log file setup**: ``setup_logging`` opens a ``.txt`` log file and
       optionally initialises a W&B run.
    10. **Evaluation loop**: iterates over ``custom_tasks`` with a tqdm outer
        progress bar. Each call to ``run_custom_task`` runs all 50 trials for
        one task and returns updated global counters + per-task metrics.
    11. **Final metrics**: computes overall success rate and constructs the
        ``all_results`` dict. Logs totals to the log file and optionally
        pushes to W&B.
    12. **Results table**: ``print_results_table`` renders the formatted
        per-task and overall summary to stdout after all evaluations complete.

    Cluster Parallelism Pattern
    ---------------------------
    To split evaluation across N nodes, launch N jobs with non-overlapping
    ``[task_start, task_end)`` ranges. For example, with 5 tasks and 5 nodes::

        node 0: --task_start 0 --task_end 1   (task 0 only)
        node 1: --task_start 1 --task_end 2   (task 1 only)
        ...
        node 4: --task_start 4 --task_end 5   (task 4 only)

    Each node writes its own log file and prints its own results table.
    Aggregate results must be manually combined after all jobs finish.

    W&B Metric Keys
    ---------------
    All metrics are prefixed by scope for clean W&B dashboard filtering:

    - ``success_rate/task_comp_{level}_overall`` : final cross-task success rate
    - ``num_episodes/task_comp_{level}_overall`` : total episodes run
    - Per-task metrics are logged by ``run_custom_task``:
      ``success_rate/{task_description}``, ``num_episodes/{task_description}``

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration object populated by draccus from CLI / YAML.

    Returns
    -------
    final_success_rate : float
        Overall success rate across all evaluated tasks in the active task
        slice, in [0.0, 1.0]. Returned so the function can be called
        programmatically (e.g. in unit tests or hyperparameter sweeps).

    Raises
    ------
    AssertionError
        If ``cfg.pretrained_checkpoint`` is empty or if both quantisation
        modes are enabled simultaneously.
    ValueError
        Propagated from ``load_custom_tasks`` if an unrecognised
        ``cfg.comp_level`` is passed (not ``"l1"`` or ``"l2"``).
    AssertionError
        Propagated from ``check_unnorm_key`` if the action un-normalisation
        key cannot be resolved in ``model.norm_stats``.
    """

    # ── Optional: block startup and wait for remote debugger attachment ───────
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))   # Listen on all interfaces, port 5678
        print("Waiting for debugger attach")
        debugpy.wait_for_client()            # Block until VS Code/PyCharm attaches

    # ── Configuration validation ──────────────────────────────────────────────
    assert cfg.pretrained_checkpoint, \
        "pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), \
        "Cannot use both 8-bit and 4-bit quantization!"

    # ── Global random seed ────────────────────────────────────────────────────
    # Set before model initialisation so weight loading is deterministic on
    # models that use random initialisation for new heads/projectors
    set_seed_everywhere(cfg.seed)

    # ── Load model and all optional components ────────────────────────────────
    model, action_head, proprio_projector, noisy_action_projector, processor = (
        initialize_model(cfg)
    )

    # ── Derive policy input image dimensions ──────────────────────────────────
    resize_size = get_image_resize_size(cfg)

    # ── Load custom task definitions and pre-recorded initial states ──────────
    all_custom_tasks = load_custom_tasks(cfg.comp_level)
    total_num_tasks  = len(all_custom_tasks)  # Total across all nodes

    # ── Select the task slice for this node ──────────────────────────────────
    # cfg.task_end == -1 is the sentinel for "run all tasks to the end"
    task_end     = cfg.task_end if cfg.task_end >= 0 else total_num_tasks
    task_start   = cfg.task_start
    custom_tasks = all_custom_tasks[task_start:task_end]  # Inclusive:Exclusive slice
    num_tasks    = len(custom_tasks)

    # ── Pre-logging to console (log file not yet open) ────────────────────────
    log_message(
        f"Loaded {total_num_tasks} total task composition "
        f"{cfg.comp_level.upper()} tasks",
        None,  # No file yet
    )
    log_message(
        f"Running task subset [{task_start}:{task_end}] ({num_tasks} tasks)",
        None,
    )
    for i, ct in enumerate(custom_tasks):
        log_message(
            f"  [{task_start + i}] {ct['task_description']} ({ct['task'].bddl_file})",
            None,
        )

    # ── Open log file and optionally initialise W&B ───────────────────────────
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Log configuration summary to the opened file
    log_message("=" * 80, log_file)
    log_message(f"TASK COMPOSITION {cfg.comp_level.upper()} EVALUATION", log_file)
    log_message(f"Model: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"Seed: {cfg.seed}", log_file)
    log_message(f"Num trials per task: {cfg.num_trials_per_task}", log_file)
    log_message(f"Num tasks: {num_tasks}", log_file)
    log_message("=" * 80, log_file)

    # ── Initialise global accumulators ────────────────────────────────────────
    total_episodes  = 0
    total_successes = 0
    task_results    = {}   # {task_description: {'success_rate': float, 'episodes': int}}

    # =========================================================================
    # MAIN EVALUATION LOOP — ITERATE OVER TASKS
    # =========================================================================

    for task_idx in tqdm.tqdm(
        range(num_tasks),
        desc=f"Task Comp {cfg.comp_level.upper()}",
    ):
        # run_custom_task handles: env creation, 50 episodes, video saving,
        # env teardown, gc.collect — returns updated global counters
        total_episodes, total_successes, task_name, task_sr, task_eps = run_custom_task(
            cfg,
            custom_tasks[task_idx],
            task_idx,
            num_tasks,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

        # Store per-task result for the results table
        task_results[task_name] = {
            'success_rate': task_sr,
            'episodes':     task_eps,
        }

    # =========================================================================
    # FINAL AGGREGATION AND REPORTING
    # =========================================================================

    # Compute overall success rate; guard against zero episodes (e.g. empty task slice)
    final_success_rate = (
        float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    )
    all_results = {
        'success_rate':    final_success_rate,
        'total_episodes':  total_episodes,
        'total_successes': total_successes,
    }

    # Log final block to the .txt file
    log_message("=" * 80, log_file)
    log_message(
        f"FINAL RESULTS - TASK COMPOSITION {cfg.comp_level.upper()}:",
        log_file,
    )
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(
        f"Overall success rate: {final_success_rate:.4f} "
        f"({final_success_rate * 100:.1f}%)",
        log_file,
    )
    log_message("=" * 80, log_file)

    # ── Optional W&B overall metric push ──────────────────────────────────────
    if cfg.use_wandb:
        import wandb
        # Key uses comp_level so L1 and L2 metrics live in separate namespaces
        wandb.log({
            f"success_rate/task_comp_{cfg.comp_level}_overall": final_success_rate,
            f"num_episodes/task_comp_{cfg.comp_level}_overall": total_episodes,
        })
        wandb.save(local_log_filepath)  # Upload the .txt log as a W&B artifact

    # ── Close log file ────────────────────────────────────────────────────────
    if log_file:
        log_file.close()

    # ── Print ASCII results table to stdout ───────────────────────────────────
    # Called after log_file.close() so the table appears only in stdout,
    # not duplicated in the log file (which already has structured logs)
    print_results_table(task_results, all_results)

    return final_success_rate


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # draccus.wrap() applied to eval_task_comp converts it into a CLI-callable
    # function. When invoked as:
    #
    #   python run_libero_eval_task_comp.py --pretrained_checkpoint /path --comp_level l1
    #
    # draccus parses sys.argv, constructs a GenerateConfig, and calls eval_task_comp(cfg).
    # The function then executes the full evaluation and returns the final_success_rate.
    #
    # When imported as a module (import run_libero_eval_task_comp), this block
    # is NOT executed, allowing programmatic use without triggering evaluation.
    eval_task_comp()