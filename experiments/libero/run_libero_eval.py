"""
run_libero_eval.py
==================

Evaluates a trained Vision-Language-Action (VLA) policy — specifically an
OpenVLA-OFT-based model — across one or more LIBERO simulation benchmark task
suites.

Overview
--------
The LIBERO benchmark is a MuJoCo/Robosuite-based simulation framework designed
to assess robotic manipulation policies under diverse task configurations.
This script drives a full evaluation loop:

    1. Parse and validate a structured configuration (via draccus).
    2. Load the VLA backbone, action head, proprioceptive projector, diffusion
       noise projector, and tokenizer/image processor.
    3. Iterate over every task in the chosen suite (or a single specified task),
       running `num_trials_per_task` independent rollouts each.
    4. Optionally replace the original task instruction with a synonym variant
       at one or more linguistic levels (l1 / l2 / l3), enabling ablation
       studies on language robustness.
    5. Log per-episode and per-task success rates to a local .txt file and,
       optionally, to Weights & Biases.
    6. Print a formatted summary table at the end of the run.

Supported Task Suites
--------------------------------------
- libero_spatial  : 10 tasks testing spatial relational reasoning
- libero_object   : 10 tasks testing object-centric manipulation
- libero_goal     : 10 tasks testing goal-conditioned manipulation
- libero_10       : 10 longer-horizon tasks (also called LIBERO-Long)
- libero_90       : 90 diverse manipulation tasks

Typical CLI Usage
--------------------------
    python run_libero_eval.py \\
        --pretrained_checkpoint /path/to/checkpoint \\
        --task_suite_name libero_goal \\
        --num_trials_per_task 50 \\
        --use_wandb True \\
        --seed 42

Key Design Decisions
--------------------
- **Open-loop action chunking**: The policy is queried once to produce a chunk
  of `num_open_loop_steps` future actions, which are executed sequentially
  before the model is re-queried. This dramatically reduces inference latency
  while maintaining high success rates.
- **Proprioceptive conditioning**: An 8-dimensional state vector
  (end-effector XYZ + axis-angle rotation + gripper positions) is projected
  into the LLM embedding space and prepended to the input token sequence.
- **Graceful action-head fallback**: If the action head weights are not found
  as a standalone file (e.g. when using a LoRA-merged checkpoint), the script
  assumes the head is fused into the model and continues without error.
- **Reproducibility**: `set_seed_everywhere` is called before each command
  level loop to guarantee identical random states across ablation conditions.

Dependencies
------------
- draccus    : Decorator-based CLI/YAML config parsing for Python dataclasses
- libero     : LIBERO benchmark environments, task definitions, and BDDL files
- wandb      : Weights & Biases experiment tracking (optional)
- numpy      : Array math, state concatenation, initial state handling
- tqdm       : Progress bars for episode and task loops
- torch      : Deep learning backend (imported lazily inside initialize_model)
- debugpy    : Remote debugger support (activated only when cfg.debug=True)
- re         : Regular expressions for BDDL variant file name matching

Author: Agostino Cardamone
"""


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import json       # Deserialise custom initial states from JSON files
import logging    # Structured, levelled console and file logging
import os         # OS-level filesystem ops: makedirs, path joins, listdir
import sys        # Manipulate sys.path to make local packages importable
import gc         # Explicit garbage collection after environment teardown

from collections import deque       # FIFO queue with a fixed maximum length,
                                    # used to store the action chunk in flight
from dataclasses import dataclass   # Declarative configuration via @dataclass
from enum import Enum               # Strongly typed, string-comparable enum
from pathlib import Path            # Object-oriented, cross-platform file paths
from typing import Optional, Union  # PEP 484 generic type annotations

import re  # POSIX-style regular expressions for matching BDDL variant filenames


# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

import draccus           # CLI wrapper that maps dataclass fields → argparse flags
import numpy as np       # Numerical arrays; used for state concatenation & seeding
import tqdm              # Wraps iterables with a real-time progress bar
from libero.libero import benchmark  # Registry of all LIBERO task suite factories

import wandb  # W&B client; initialised conditionally based on cfg.use_wandb


# =============================================================================
# PROJECT ROOT RESOLUTION
# =============================================================================
# This block ensures that sibling top-level packages (experiments/, prismatic/)
# are importable regardless of the directory from which the script is invoked.

from pathlib import Path

current_file = Path(__file__).resolve()           # Absolute path of this script
project_root = current_file.parent.parent.parent  # Three levels up → repo root
sys.path.insert(0, str(project_root))             # Prepend to sys.path so local
                                                   # packages shadow any installed
                                                   # versions of the same name


# =============================================================================
# LIBERO PATH UTILITY
# =============================================================================

from libero.libero import get_libero_path  # Returns the absolute path to a named
                                            # LIBERO resource directory, e.g.
                                            # get_libero_path("bddl_files")


# =============================================================================
# INTERNAL UTILITIES — LIBERO-SPECIFIC HELPERS
# =============================================================================

from experiments.libero.utils.libero_utils import (
    get_libero_dummy_action,   # Returns a neutral (zero) action for warm-up steps,
                               # formatted correctly for the given model family
    get_libero_env,            # Instantiates a LIBERO/Robosuite MuJoCo environment
                               # for a given task, optionally overriding the BDDL file
    get_libero_image,          # Extracts the third-person RGB camera frame from an
                               # observation dictionary as a NumPy (H,W,3) array
    get_libero_wrist_image,    # Extracts the wrist-mounted camera RGB frame from an
                               # observation dictionary as a NumPy (H,W,3) array
    quat2axisangle,            # Converts a 4D quaternion [qx,qy,qz,qw] to a 3D
                               # axis-angle vector [ax,ay,az]; used for proprioception
    save_rollout_video,        # Renders a list of RGB frames to an MP4 video file
                               # and optionally logs it to W&B
)


# =============================================================================
# INTERNAL UTILITIES — OPENVLA MODEL COMPONENTS
# =============================================================================

from experiments.openvla_utils import (
    get_action_head,            # Loads the continuous action prediction head from disk;
                                # supports L1-regression or DDIM-diffusion variants
    get_noisy_action_projector, # Loads the MLP that injects diffusion-step noise
                                # conditioning into the action head during DDIM inference
    get_processor,              # Returns the VLA tokenizer + image pre-processor
                                # (a HuggingFace ProcessorMixin subclass)
    get_proprio_projector,      # Loads the MLP that maps the raw proprioceptive state
                                # vector into the LLM's hidden embedding dimension
    resize_image_for_policy,    # Resizes a raw NumPy image (H,W,3) to the (H',W')
                                # resolution expected by the VLA vision encoder
)


# =============================================================================
# INTERNAL UTILITIES — GENERAL ROBOT EXPERIMENT HELPERS
# =============================================================================

from experiments.robot_utils import (
    DATE_TIME,                # Module-level string constant: ISO-formatted timestamp
                               # captured at import time, e.g. "20260416_105400"
    get_image_resize_size,    # Reads cfg and returns the (H, W) tuple to which
                               # policy input images must be resized
    invert_gripper_action,    # Flips the sign of the gripper dimension of an action
                               # vector to reconcile OpenVLA's training convention
                               # (0=close, 1=open) with the env convention (-1=open)
    normalize_gripper_action, # Linearly maps the gripper dimension from [0, 1]
                               # to [-1, +1] as required by the LIBERO env step()
    set_seed_everywhere,      # Sets Python random, NumPy, and PyTorch seeds in one
                               # call to ensure full reproducibility
)


# =============================================================================
# VLA INFERENCE ENTRY POINTS
# =============================================================================

from experiments.openvla_utils import (
    get_vla,        # Instantiates the complete VLA model (vision encoder + LLM)
                    # from a local or HuggingFace checkpoint; applies quantisation
                    # and LoRA adapters as specified in cfg
    get_vla_action, # Runs a single forward pass of the VLA model and returns an
                    # action chunk of shape (num_open_loop_steps, action_dim)
)

from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # Integer constant defining
                                                        # the expected action chunk
                                                        # size (typically 8) [web:3]


# =============================================================================
# TASK SUITE ENUMERATION
# =============================================================================
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    
    #################################################################################################################
    # Command variation parameters
    #################################################################################################################
    change_command: bool = False                     # Use synonym command variations
    command_level: Optional[str] = None              # Command level: 'l1', 'l2', 'l3', 'all', or None
    only_numbered_variants: bool = False              #if true, skip "base" and test only v1, v2, v3...
    selected_version: Optional[Union[int, str]] = None  # e.g. 1,2,3 or "base"
    
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL  # Task suite
    task_id: Optional[int] = None                    # Specific task ID to evaluate (0-indexed). If None, evaluate all tasks
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "/mnt/beegfs/a.cardamone7/outputs/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 42                                    # Random Seed (for reproducibility)

    debug: bool = False  
    # fmt: on



# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(cfg: GenerateConfig) -> None:
    """
    Assert that the evaluation configuration is internally consistent.

    This function acts as a pre-flight check: all assertions run before any
    model is loaded or environment is created, so misconfiguration produces
    a clear, actionable error message rather than a cryptic downstream failure.

    Validation Rules
    ----------------
    1. ``pretrained_checkpoint`` must be explicitly set (not ``None``).
    2. If ``"image_aug"`` appears in the checkpoint path, ``center_crop`` must
       be ``True``. Models trained with random-crop augmentation expect a
       deterministic center crop at inference time to avoid distribution shift.
    3. ``load_in_8bit`` and ``load_in_4bit`` are mutually exclusive; enabling
       both simultaneously would produce undefined quantisation behaviour.
    4. ``task_suite_name`` must be one of the five values registered in
       ``TaskSuite`` to prevent downstream KeyErrors in ``TASK_MAX_STEPS`` and
       the LIBERO benchmark registry.

    Parameters
    ----------
    cfg : GenerateConfig
        The fully-populated configuration object to validate.

    Returns
    -------
    None
        Returns silently if all assertions pass.

    Raises
    ------
    AssertionError
        Raised immediately on the first failing check, with a descriptive
        human-readable message pinpointing the misconfiguration.
    """
    # Rule 1: Checkpoint must be provided
    assert cfg.pretrained_checkpoint is not None, \
        "pretrained_checkpoint must not be None!"

    # Rule 2: Image-augmented checkpoints require center cropping at eval time
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, (
            "Expecting `center_crop==True` because model was trained with "
            "image augmentations!"
        )

    # Rule 3: 8-bit and 4-bit quantisation are mutually exclusive
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), \
        "Cannot use both 8-bit and 4-bit quantization!"

    # Rule 4: Task suite name must be a registered enum value
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], \
        f"Invalid task suite: {cfg.task_suite_name}"


# =============================================================================
# MODEL INITIALISATION
# =============================================================================

def initialize_model(cfg: GenerateConfig):
    """
    Instantiate all neural network components required for policy inference.

    Each component is loaded conditionally based on the architecture flags in
    ``cfg``, minimising GPU memory usage when optional components are disabled.
    CUDA memory is explicitly cleared before loading to reduce fragmentation
    from any previously allocated tensors.

    Architecture Overview
    ---------------------
    The full inference graph is:

        raw image(s) → resize → VLA backbone → LLM hidden states
                                                      ↓
        raw proprio  → proprio_projector      →  concat
                                                      ↓
                                              action_head
                                           (L1 or diffusion)
                                                      ↓
                                              action chunk

    When diffusion is used, the noisy_action_projector additionally injects
    noise-step conditioning into the action head at each DDIM denoising step.

    Components
    ----------
    model : torch.nn.Module
        The full VLA model (visual encoder + language model backbone).
        Always loaded. Quantisation (8-bit / 4-bit) is applied here if enabled.

    proprio_projector : torch.nn.Module or None
        A small MLP mapping the 8-dimensional proprioceptive state vector to
        the LLM's ``llm_dim``-dimensional embedding space. Loaded only when
        ``cfg.use_proprio=True``. The 8 dimensions are:
        - 3: end-effector position (x, y, z)
        - 3: end-effector orientation as axis-angle (ax, ay, az)
        - 2: gripper joint positions (left, right)

    action_head : torch.nn.Module or None
        Continuous action prediction head. Loaded when ``cfg.use_l1_regression``
        or ``cfg.use_diffusion`` is ``True``. If the standalone weight file is
        not found (e.g. when the head is fused into a LoRA checkpoint), the
        function logs a warning and returns ``None`` — the head is then expected
        to be embedded within ``model``.

    noisy_action_projector : torch.nn.Module or None
        Noise conditioning projector for DDIM diffusion. Loaded only when
        ``cfg.use_diffusion=True``. Maps the current diffusion time step (as a
        scalar or embedding) into the action head's conditioning space.

    processor : transformers.ProcessorMixin or None
        Tokenizer and image pre-processor for the VLA model. Loaded only when
        ``cfg.model_family == "openvla"``. After loading, ``check_unnorm_key``
        is called to resolve and validate the action un-normalisation key.

    Parameters
    ----------
    cfg : GenerateConfig
        Evaluation configuration containing checkpoint paths, architecture
        flags, and quantisation settings.

    Returns
    -------
    model : torch.nn.Module
        VLA backbone in evaluation mode (gradients disabled).
    action_head : torch.nn.Module or None
        Continuous action head, or ``None`` if not applicable / not found.
    proprio_projector : torch.nn.Module or None
        Proprioceptive state projector, or ``None`` if not used.
    noisy_action_projector : torch.nn.Module or None
        Diffusion noise projector, or ``None`` if not used.
    processor : transformers.ProcessorMixin or None
        VLA input processor, or ``None`` if not applicable.
    """
    import torch  # Imported lazily to avoid mandatory GPU dependency at module level

    # Free cached GPU tensors and wait for all CUDA kernels to finish,
    # reducing the risk of OOM errors when loading large model weights.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # Release the CUDA memory cache
        torch.cuda.synchronize()   # Block until all CUDA operations complete

    # ── 1. Load the VLA backbone (always required) ───────────────────────────
    model = get_vla(cfg)

    # ── 2. Proprioceptive projector (optional) ────────────────────────────────
    proprio_projector = None           # Default: no proprio conditioning
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,   # Must match the LLM hidden dimension for alignment
            proprio_dim=8,   # Fixed 8-dim proprio vector for all LIBERO tasks
        )

    # ── 3. Continuous action head (optional, with graceful fallback) ──────────
    action_head = None                 # Default: no separate action head
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            logger.info("Action head loaded as a separate file.")
        except (AssertionError, FileNotFoundError):
            # The action head is not a standalone file — it has been merged into
            # the LoRA checkpoint weights. Log a warning and continue.
            logger.warning("Action head not found as a separate file.")
            logger.warning("Assuming it is embedded in the model (LoRA checkpoint).")
            action_head = None

    # ── 4. Diffusion noise projector (optional, only for DDIM) ───────────────
    noisy_action_projector = None      # Default: no diffusion noise conditioning
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # ── 5. Input processor (only for OpenVLA family) ──────────────────────────
    processor = None                   # Default: no processor
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # Validate that the action un-norm key exists in the loaded model and
        # write the resolved key back into cfg for downstream use.
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


# =============================================================================
# ACTION UN-NORMALISATION KEY RESOLUTION
# =============================================================================

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """
    Resolve and validate the action un-normalisation key in the VLA model.

    During training, the dataset statistics (per-dimension mean and standard
    deviation) for all action components are stored inside the model checkpoint
    under ``model.norm_stats``, a dictionary keyed by dataset name. At
    inference time, the raw predicted action values must be un-normalised
    (i.e. de-standardised) using those same statistics before being sent to
    the robot or simulator.

    This function identifies the correct key for the current task suite,
    writes it into ``cfg.unnorm_key`` (side-effect), and raises a clear error
    if no valid key can be found.

    Key Resolution Order
    --------------------
    1. If ``cfg.unnorm_key`` is already a non-empty string **and** that string
       exists in ``model.norm_stats``, use it as-is (respects user override).
    2. Otherwise, start from ``cfg.task_suite_name`` as the candidate key.
    3. If the candidate is absent from ``norm_stats`` but the suffixed variant
       ``"{candidate}_no_noops"`` is present, adopt the suffixed key. The
       ``_no_noops`` suffix indicates the dataset was filtered to remove no-op
       (null-action) time steps before training.
    4. If neither the plain nor ``_no_noops`` key exists, raise
       ``AssertionError`` with the candidate key name for easy debugging.

    Side Effects
    ------------
    Mutates ``cfg.unnorm_key`` in place with the resolved key string.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration object. ``cfg.unnorm_key`` may be updated.
    model : torch.nn.Module
        The loaded VLA model. Must expose a ``norm_stats`` dict attribute
        whose keys are dataset/suite names.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If no valid un-norm key can be located in ``model.norm_stats``.
    """
    # Log all available keys and the current config value for debugging
    logger.info(f"Available norm_stats keys: {list(model.norm_stats.keys())}")
    logger.info(f"cfg.unnorm_key from config: '{cfg.unnorm_key}'")

    # Priority 1: Honour an explicitly set, valid user override
    if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
        logger.info(f"Using user-specified unnorm_key: {cfg.unnorm_key}")
        return  # Nothing more to do

    # Priority 2: Derive the candidate key from the task suite name
    unnorm_key = cfg.task_suite_name

    # Priority 3: Try the "_no_noops" variant if the plain key is absent
    if (unnorm_key not in model.norm_stats
            and f"{unnorm_key}_no_noops" in model.norm_stats):
        unnorm_key = f"{unnorm_key}_no_noops"  # Adopt the no-noops variant

    # Fail with a descriptive message if neither key exists
    assert unnorm_key in model.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in VLA `norm_stats`!"
    )

    # Persist the resolved key back into the configuration object
    cfg.unnorm_key = unnorm_key


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(cfg: GenerateConfig):
    """
    Initialise per-run logging to a local file and optionally to W&B.

    A unique, human-readable run ID is constructed from the task suite name,
    model family, and a timestamp, with optional suffixes for the command
    variation level and user-provided notes. This run ID is used as both the
    log file name and the W&B run name, making it easy to correlate artefacts.

    Run ID Format
    -------------
    ``EVAL-{task_suite_name}-{model_family}-{DATE_TIME}[--{run_id_note}][--{command_level}]``

    Example: ``EVAL-libero_goal-openvla-20260416_105400--ablation_v2--l1``

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration object providing task suite name, model family,
        optional run note, W&B settings, and local log directory.

    Returns
    -------
    log_file : io.TextIOWrapper
        An open file handle (write mode) for the ``.txt`` log file.
        Callers are responsible for closing it when the evaluation is done.
    local_log_filepath : str
        Absolute path of the ``.txt`` log file that was created.
    run_id : str
        The unique run identifier string used for both the log file name
        and (if enabled) the W&B run name.
    """
    # ── Build the unique run ID ───────────────────────────────────────────────
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"

    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"     # Append user-provided annotation

    if cfg.change_command and cfg.command_level:
        run_id += f"--{cfg.command_level}"   # Append the command variation level

    # ── Set up local text log file ────────────────────────────────────────────
    os.makedirs(cfg.local_log_dir, exist_ok=True)  # Create log dir if missing
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")        # Open for writing (truncates any existing file)
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # ── Optionally initialise Weights & Biases ────────────────────────────────
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,    # W&B user/team
            project=cfg.wandb_project,  # W&B project board
            name=run_id,                # Human-readable run name in the W&B UI
        )

    return log_file, local_log_filepath, run_id


# =============================================================================
# DUAL-SINK LOG HELPER
# =============================================================================

def log_message(message: str, log_file=None) -> None:
    """
    Write a message to the module logger and optionally to a text log file.

    This helper ensures that every important status update appears both in the
    console (via the Python ``logging`` infrastructure) and in the persistent
    per-run ``.txt`` log file. Using ``flush()`` after each write guarantees
    that the file is up-to-date even if the process terminates unexpectedly.

    Parameters
    ----------
    message : str
        The text to log. Should not contain a trailing newline; one is added
        automatically when writing to the file.
    log_file : io.TextIOWrapper or None, optional
        An open writable file handle. When ``None``, the message is only sent
        to the console logger.

    Returns
    -------
    None
    """
    logger.info(message)           # Always log to console at INFO level
    if log_file:
        log_file.write(message + "\n")  # Append a newline for line-by-line readability
        log_file.flush()               # Force OS write-through to avoid buffering


# =============================================================================
# INITIAL STATE LOADING
# =============================================================================

def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """
    Load the initial simulator states for a given task.

    Each LIBERO rollout begins from a fixed initial state (object positions,
    joint angles, etc.) drawn from a pre-recorded set. This function supports
    two sources:

    1. **Default LIBERO states** (``cfg.initial_states_path == "DEFAULT"``):
       The benchmark provides 50 reference initial states per task. These
       correspond to diverse, challenging starting configurations and are the
       standard for reproducible benchmarking [web:9].

    2. **Custom states from a JSON file**: When ``cfg.initial_states_path``
       points to a JSON file, that file is loaded and returned alongside the
       default states. This mode is used to evaluate the policy on a curated
       subset of states — for example, only states where an expert demonstration
       was successful.

    JSON File Format
    ----------------
    The JSON file is expected to be a nested dictionary with the following
    structure::

        {
          "<task_description_with_underscores>": {
            "demo_0": {"success": true,  "initial_state": [...]},
            "demo_1": {"success": false, "initial_state": [...]},
            ...
          },
          ...
        }

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration providing ``initial_states_path``.
    task_suite : libero.libero.benchmark.TaskSuite
        The instantiated LIBERO task suite object, used to retrieve the
        default initial states for this task.
    task_id : int
        0-indexed integer ID of the task within the suite.
    log_file : io.TextIOWrapper or None, optional
        Open log file handle for status messages.

    Returns
    -------
    initial_states : list
        The default LIBERO initial states for this task (always returned).
    all_initial_states : dict or None
        The full parsed JSON dictionary when a custom path is given, or
        ``None`` when using default states.
    """
    # Always load the default states bundled with the benchmark
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        # Open and parse the user-supplied JSON file
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states   # Return both default + custom
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None                 # No custom states


# =============================================================================
# OBSERVATION PREPARATION
# =============================================================================

def prepare_observation(obs: dict, resize_size: tuple) -> tuple:
    """
    Convert a raw LIBERO observation dictionary into the format expected by
    the VLA model.

    LIBERO observations are raw dictionaries from the Robosuite simulator.
    The VLA model requires:
    - Resized RGB images from two camera views (third-person + wrist).
    - A concatenated 8-dimensional proprioceptive state vector.

    Proprioceptive State Composition
    ---------------------------------
    The 8-dimensional state vector is formed by concatenating:
    - ``obs["robot0_eef_pos"]``       : (3,) end-effector XYZ position in world frame
    - ``quat2axisangle(obs["robot0_eef_quat"])`` : (3,) end-effector orientation
      converted from quaternion (qx, qy, qz, qw) to axis-angle (ax, ay, az)
    - ``obs["robot0_gripper_qpos"]``  : (2,) left and right gripper joint positions

    Why axis-angle instead of quaternion? Axis-angle provides a minimal (3D)
    continuous representation of rotation, avoiding the redundancy and
    sign ambiguity of quaternions, which simplifies learning for the MLP
    proprio projector.

    Parameters
    ----------
    obs : dict
        Raw observation dictionary from ``env.step()`` or ``env.reset()``.
        Required keys: ``"robot0_eef_pos"``, ``"robot0_eef_quat"``,
        ``"robot0_gripper_qpos"``, plus the camera image keys consumed
        internally by ``get_libero_image`` and ``get_libero_wrist_image``.
    resize_size : tuple of (int, int)
        Target (height, width) in pixels to resize policy input images to.

    Returns
    -------
    observation : dict
        Processed observation with keys:
        - ``"full_image"``  : np.ndarray (H', W', 3) third-person view, resized
        - ``"wrist_image"`` : np.ndarray (H', W', 3) wrist view, resized
        - ``"state"``       : np.ndarray (8,) proprioceptive state vector
    img : np.ndarray
        The original (un-resized) third-person camera frame for video replay.
    """
    # ── Extract raw camera frames from the observation dict ───────────────────
    img = get_libero_image(obs)            # Third-person RGB frame (H, W, 3)
    wrist_img = get_libero_wrist_image(obs)  # Wrist-mounted RGB frame (H, W, 3)

    # ── Resize images to the resolution expected by the VLA backbone ──────────
    img_resized       = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # ── Assemble the observation dictionary consumed by get_vla_action ────────
    observation = {
        "full_image":  img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate((
            obs["robot0_eef_pos"],                         # (3,) XYZ position
            quat2axisangle(obs["robot0_eef_quat"]),        # (3,) axis-angle rotation
            obs["robot0_gripper_qpos"],                    # (2,) gripper joint angles
        )),  # → shape (8,)
    }

    # Return the processed observation dict AND the original full image.
    # The original (un-resized) image is used for video replay recording.
    return observation, img


# =============================================================================
# ACTION POST-PROCESSING
# =============================================================================

def process_action(action: np.ndarray, model_family: str) -> np.ndarray:
    """
    Post-process a raw predicted action before sending it to the simulator.

    Two sequential transformations are applied to reconcile the conventions
    used during training with those expected by the LIBERO environment:

    Transformation 1 — Gripper normalisation
    -----------------------------------------
    The VLA action head outputs a gripper dimension in [0, 1] (continuous) or
    as a binary value. The LIBERO ``env.step()`` function expects the gripper
    dimension to be in [-1, +1], where -1 means fully open and +1 means fully
    closed. ``normalize_gripper_action`` performs this remapping, and the
    ``binarize=True`` flag additionally snaps the output to exactly -1 or +1.

    Transformation 2 — OpenVLA sign inversion
    -------------------------------------------
    During the OpenVLA training data pipeline, the gripper dimension sign is
    flipped to harmonise the convention across heterogeneous robot datasets:
        training convention: 0 → open,  1 → close  (after normalisation: -1 → open)
        LIBERO env convention: -1 → open, +1 → close
    After Transformation 1, the sign is already correct for LIBERO, BUT because
    the dataloader flipped it *during training*, the model learned to predict with
    the flipped convention. ``invert_gripper_action`` restores the original sign so
    the environment receives the correct command [web:3].

    Parameters
    ----------
    action : np.ndarray of shape (action_dim,)
        A single action vector as predicted by the model. The last dimension
        is assumed to be the gripper control signal.
    model_family : str
        The model family identifier (e.g. ``"openvla"``). The sign inversion
        is only applied for OpenVLA models; other families may have different
        conventions.

    Returns
    -------
    action : np.ndarray of shape (action_dim,)
        The post-processed action vector ready for ``env.step()``.
    """
    # Step 1: Remap gripper from [0,1] → {-1, +1} (binarised)
    action = normalize_gripper_action(action, binarize=True)

    # Step 2: Flip gripper sign to undo the training-time inversion
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action
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
    Execute a single evaluation episode in the LIBERO simulator.

    The episode loop implements **open-loop action chunking**: the policy is
    queried once to produce a chunk of ``cfg.num_open_loop_steps`` actions,
    which are then executed sequentially in the environment. Once the chunk is
    exhausted, the model is re-queried with the latest observation. This
    strategy dramatically reduces inference overhead compared to querying the
    model at every step, while maintaining high task performance [web:7].

    Episode Lifecycle
    -----------------
    1. **Reset**: The environment is reset to a clean state, then
       ``initial_state`` is applied (if provided) via ``env.set_init_state``.
    2. **Warm-up** (steps 0 … ``cfg.num_steps_wait - 1``): Dummy (zero) actions
       are applied to allow objects to settle under gravity. No observations
       are stored and the policy is not queried during this phase.
    3. **Policy execution** (steps ``cfg.num_steps_wait`` … ``max_steps``):
       - Observation is prepared via ``prepare_observation``.
       - If the action queue is empty, the model is queried for a new action
         chunk via ``get_vla_action``.
       - The front action is dequeued, post-processed via ``process_action``,
         and sent to the environment.
       - If the environment signals ``done=True``, the episode succeeds and
         terminates early.
    4. **Error handling**: Any exception during execution is caught, logged,
       and the episode is marked as failed (``success=False``).

    Action Queue Mechanics
    ----------------------
    ``action_queue`` is a ``deque`` with ``maxlen=cfg.num_open_loop_steps``.
    When ``extend(actions)`` is called with a full chunk, the deque fills up.
    ``popleft()`` extracts actions FIFO until the deque is empty, at which
    point the model is re-queried. The ``maxlen`` cap ensures the queue never
    grows beyond one chunk.

    Parameters
    ----------
    cfg : GenerateConfig
        Evaluation configuration (step limits, model family, etc.).
    env : robosuite.environments.base.MujocoEnv
        The instantiated LIBERO/Robosuite simulation environment.
    task_description : str
        Natural language task instruction passed to the VLA as conditioning.
    model : torch.nn.Module
        The loaded VLA backbone.
    resize_size : tuple of (int, int)
        Target resolution (H, W) for policy input images.
    processor : transformers.ProcessorMixin or None
        VLA tokenizer/image processor (None if not applicable).
    action_head : torch.nn.Module or None
        Continuous action head (None if embedded in model or not used).
    proprio_projector : torch.nn.Module or None
        Proprioceptive state projector (None if not used).
    noisy_action_projector : torch.nn.Module or None
        Diffusion noise projector (None if not used).
    initial_state : np.ndarray or None
        Pre-recorded simulator state to restore at episode start. If ``None``,
        the environment's default reset state is used.
    log_file : io.TextIOWrapper or None
        Open log file handle for error messages.

    Returns
    -------
    success : bool
        ``True`` if the environment signalled task completion (``done=True``)
        before the step budget was exhausted; ``False`` otherwise.
    replay_images : list of np.ndarray
        Ordered list of raw (un-resized) third-person camera frames captured
        at each active step (warm-up frames excluded). Used to render a
        video replay of the episode.
    replay_states : list of np.ndarray
        Ordered list of end-effector (x, y, z) positions captured at each
        active step, useful for trajectory visualisation and analysis.
    """
    # ── Phase 1: Environment reset ────────────────────────────────────────────
    env.reset()  # Full reset: randomise scene, reload assets

    if initial_state is not None:
        # Restore a specific pre-recorded simulator state (deterministic start)
        obs = env.set_init_state(initial_state)
    else:
        # Use the default post-reset observation (stochastic start)
        obs = env.get_observation()

    # ── Initialise action chunk queue ─────────────────────────────────────────
    # maxlen ensures the deque never grows beyond one chunk size
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # ── Episode state variables ───────────────────────────────────────────────
    t = 0              # Current simulator step counter
    replay_images = [] # Accumulates third-person frames for video replay
    replay_states = [] # Accumulates end-effector XYZ positions for analysis
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]  # Hard step budget for this suite

    # ── Phase 2 & 3: Main episode loop ───────────────────────────────────────
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # ── Phase 2: Warm-up (dummy actions, objects settle) ──────────────
            if t < cfg.num_steps_wait:
                # Apply a zero/neutral action; discard the resulting observation
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue  # Skip observation preparation and policy querying

            # ── Phase 3: Active policy execution ──────────────────────────────

            # Prepare model input from the current raw observation
            observation, img = prepare_observation(obs, resize_size)

            replay_images.append(img)                       # Save raw frame for video
            replay_states.append(obs["robot0_eef_pos"])     # Save EEF XYZ for trajectory

            # Re-query the model if the action chunk has been fully consumed
            if len(action_queue) == 0:
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
                )  # → list of num_open_loop_steps action vectors

                # Debug: print the first action to verify the model is producing varied outputs
                if t < 50:  # Only log during the first 50 active steps
                    logger.info(f"Step {t}: action[0] = {actions[0][:4]}...")  # First 4 of 7 dims

                action_queue.extend(actions)  # Fill the queue with the new chunk

            # Dequeue the next action (FIFO order preserves temporal coherence)
            action = action_queue.popleft()

            # Post-process: gripper normalisation + sign convention correction
            action = process_action(action, cfg.model_family)

            # Send action to simulator; receive next observation and task signal
            obs, reward, done, info = env.step(action.tolist())

            if done:
                success = True  # Environment confirmed task completion
                break

            t += 1  # Advance step counter only for active (non-warm-up) steps

    except Exception as e:
        # Catch all exceptions (e.g. simulator crashes, NaN actions) and
        # log them without re-raising so the evaluation loop can continue.
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images, replay_states

# =============================================================================
# SINGLE-TASK EVALUATION RUNNER
# =============================================================================

def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
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
    Run the full evaluation protocol for a single LIBERO task, iterating over
    all applicable command variant versions and all trial episodes.

    This is the central orchestration function of the evaluation loop. For each
    task it:

    1. Resolves which BDDL variant files (command synonym versions) exist on
       disk for the selected linguistic level.
    2. Determines which of those variants to test based on config flags.
    3. For each variant, instantiates a fresh LIBERO environment, runs
       ``cfg.num_trials_per_task`` episodes via ``run_episode``, accumulates
       success counts, saves replay videos, and logs results.
    4. Tears down the environment and runs garbage collection after each
       variant to free MuJoCo resources.
    5. Aggregates results across all variants and prints a per-task summary.

    BDDL Variant Discovery
    ----------------------
    When ``cfg.change_command=True`` and ``cfg.command_level`` is set, the
    function scans the BDDL directory for the task and collects:

    - **Base variant**: a file named ``{task_base_name}_syn_{level}.bddl``
      (e.g. ``pick_up_the_mug_syn_l1.bddl``). Tagged as ``"base"``.
    - **Numbered variants**: files matching the pattern
      ``{task_base_name}_syn_{level}_v{N}.bddl`` where N is a positive integer
      (e.g. ``pick_up_the_mug_syn_l1_v1.bddl``). Tagged with their integer N.

    After discovery, variants are sorted with ``"base"`` first, then
    ``v1, v2, v3, …`` in ascending order.

    Version Selection Logic
    -----------------------
    The set of variants actually tested (``versions_to_test``) is determined
    by the following priority order:

    1. If ``cfg.selected_version`` is set → test that single version only.
    2. Else if variants were found and ``cfg.only_numbered_variants=True`` →
       exclude ``"base"``; test only integer-numbered variants. Falls back to
       all variants (including base) if no numbered variants exist.
    3. Else if variants were found → test all discovered variants.
    4. Else (no variants found, or command variation disabled) → test the
       default environment with the original task description (``[None]``).

    Per-Version Episode Loop
    ------------------------
    For each version, the function iterates over ``cfg.num_trials_per_task``
    episodes. The initial state for each episode is selected as follows:

    - If ``cfg.initial_states_path == "DEFAULT"``: use the LIBERO-bundled
      state at index ``episode_idx``.
    - Otherwise: look up the state in the custom JSON dict using
      ``task_description_with_underscores/demo_{episode_idx}/initial_state``.
      If that demo is marked as failed (``"success": false``), the episode is
      **skipped** to ensure the policy is only evaluated on states that were
      reachable by the expert.

    Environment Lifecycle
    ---------------------
    A new ``env`` is created via ``get_libero_env`` for each variant (not each
    episode). After all episodes of a variant are done, ``env.close()`` is
    called inside a ``try/except`` block (some LIBERO environment versions do
    not implement ``close()``), followed by ``gc.collect()`` to ensure MuJoCo
    resources are freed before the next variant is initialised.

    Parameters
    ----------
    cfg : GenerateConfig
        Full evaluation configuration.
    task_suite : libero.libero.benchmark.TaskSuite
        The instantiated LIBERO task suite object providing task metadata.
    task_id : int
        0-indexed integer ID of the task within the suite.
    model : torch.nn.Module
        The loaded VLA backbone in eval mode.
    resize_size : tuple of (int, int)
        Target (H, W) resolution for policy input images.
    processor : transformers.ProcessorMixin or None
        VLA tokenizer/image processor; ``None`` if not applicable.
    action_head : torch.nn.Module or None
        Continuous action prediction head; ``None`` if embedded in model.
    proprio_projector : torch.nn.Module or None
        Proprioceptive state projector; ``None`` if not used.
    noisy_action_projector : torch.nn.Module or None
        DDIM diffusion noise projector; ``None`` if not used.
    total_episodes : int, default 0
        Running count of all episodes evaluated so far across tasks.
        Updated in place and returned so the caller can accumulate totals.
    total_successes : int, default 0
        Running count of all successful episodes across tasks.
        Updated in place and returned so the caller can accumulate totals.
    log_file : io.TextIOWrapper or None
        Open log file handle for structured status messages.

    Returns
    -------
    total_episodes : int
        Updated total episode count (input + episodes run in this call).
    total_successes : int
        Updated total success count (input + successes in this call).
    task_description : str
        The task language description used in the last tested variant.
        Used by the caller to key the per-task results dictionary.
    task_success_rate : float
        Aggregate success rate across all variants and episodes for this task.
        Computed as ``total_task_successes / total_task_episodes``.
    total_task_episodes : int
        Total number of episodes run for this task across all variants.

    Raises
    ------
    AssertionError
        If ``cfg.selected_version`` is set but ``cfg.change_command=False``
        or ``cfg.command_level=None``, since a specific variant requires an
        active command variation mode.
    """

    # ── Retrieve task metadata from the suite ────────────────────────────────
    # task exposes: task.bddl_file (path), task.problem_folder, task.language
    task = task_suite.get_task(task_id)

    # Load default LIBERO initial states (and optionally custom JSON states)
    initial_states, all_initial_states = load_initial_states(
        cfg, task_suite, task_id, log_file
    )

    # ── Guard: selected_version requires command variation to be active ───────
    if cfg.selected_version is not None:
        assert cfg.change_command and cfg.command_level is not None, (
            f"selected_version={cfg.selected_version} requires "
            f"change_command=True and command_level to be non-None."
        )

    # =========================================================================
    # BDDL VARIANT DISCOVERY
    # =========================================================================

    available_variants = []  # List of (version_tag, filename) tuples

    if cfg.change_command and cfg.command_level is not None:

        # Derive the base name of the task's BDDL file (without extension)
        # e.g. "pick_up_the_mug_from_the_plate" from "pick_up_the_mug_from_the_plate.bddl"
        base_name = os.path.splitext(os.path.basename(task.bddl_file))[0]

        # Resolve the directory containing BDDL files for this task
        try:
            # Primary approach: use the LIBERO path registry
            bddl_folder = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder
            )
        except Exception:
            # Fallback: use the directory of the task's own BDDL file
            bddl_folder = os.path.dirname(task.bddl_file)

        # Build the expected filename for the base (un-versioned) synonym file
        # e.g. "pick_up_the_mug_from_the_plate_syn_l1.bddl"
        base_variant_filename = f"{base_name}_syn_{cfg.command_level}.bddl"

        try:
            for filename in os.listdir(bddl_folder):

                # Skip non-BDDL files (e.g. README, JSON, hidden files)
                if not filename.endswith(".bddl"):
                    continue

                # ── Match the base synonym file (e.g. task_syn_l1.bddl) ──────
                if filename.lower() == base_variant_filename.lower():
                    available_variants.append(("base", filename))
                    continue  # No need to check version pattern for base file

                # ── Match numbered variant files (e.g. task_syn_l1_v3.bddl) ──
                # Pattern: exactly {base_name}_syn_{level}_v{digits}.bddl
                version_pattern = (
                    rf"^{re.escape(base_name)}_syn_"
                    rf"{re.escape(cfg.command_level)}_v(\d+)\.bddl$"
                )
                match = re.match(version_pattern, filename, re.IGNORECASE)
                if match:
                    version_num = int(match.group(1))  # Extract N from "_vN"
                    available_variants.append((version_num, filename))

        except Exception as e:
            # Non-fatal: log and continue with whatever variants were found
            log_message(f"Warning: Could not list variant files: {e}", log_file)

        # Sort variants: "base" gets a sort key of -1 (comes first),
        # then integer versions sort in ascending numerical order (v1, v2, v3, …)
        available_variants.sort(
            key=lambda x: (-1 if x[0] == "base" else x[0])
        )

    # =========================================================================
    # VERSION SELECTION
    # =========================================================================

    if cfg.selected_version is not None:
        # User explicitly requested a single specific version
        versions_to_test = [cfg.selected_version]

    elif available_variants:
        if cfg.only_numbered_variants:
            # Exclude the un-versioned "base" file; keep only v1, v2, v3, …
            versions_to_test = [v[0] for v in available_variants if v[0] != "base"]

            if not versions_to_test:
                # Edge case: only the base file exists — log a warning and fall back
                log_message(
                    "WARNING: only_numbered_variants=True but no numbered variants found. "
                    "Falling back to 'base'.",
                    log_file,
                )
                versions_to_test = [v[0] for v in available_variants]  # Include base
        else:
            # Test all discovered variants (base + v1 + v2 + …)
            versions_to_test = [v[0] for v in available_variants]

    else:
        # No variant files found or command variation disabled: run with
        # the default environment (original task description unchanged)
        versions_to_test = [None]

    # Log task header
    log_message("=" * 80, log_file)
    log_message(f"TASK {task_id + 1}/{task_suite.n_tasks}", log_file)
    log_message(f"Versions to test: {versions_to_test}", log_file)
    log_message("=" * 80, log_file)

    # Dictionary to accumulate per-version statistics for the final summary
    task_results_per_version = {}  # {version_label: {'success_rate', 'episodes', 'successes'}}

    # =========================================================================
    # OUTER LOOP: ITERATE OVER COMMAND VARIANT VERSIONS
    # =========================================================================

    for version_to_test in versions_to_test:

        # ── Resolve the BDDL file path for this specific variant ──────────────
        ablation_bddl_file = None  # Default: use the original task BDDL file
        if available_variants and version_to_test is not None:
            # Find the filename corresponding to this version tag
            selected_files = [
                v[1] for v in available_variants if v[0] == version_to_test
            ]
            if selected_files:
                ablation_bddl_file = selected_files[0]  # Take the first (and only) match

        # ── Instantiate the LIBERO environment for this variant ───────────────
        # get_libero_env returns:
        #   env              : the MuJoCo environment object
        #   task_description : the (possibly modified) language instruction
        #   original_description : the original unmodified BDDL instruction
        env, task_description, original_description = get_libero_env(
            task,
            change_command=cfg.change_command,
            command_level=cfg.command_level,
            ablation_bddl_file=ablation_bddl_file,  # None → use original BDDL
            resolution=cfg.env_img_res,              # MuJoCo render resolution
        )

        # ── Build a human-readable version label for logging ──────────────────
        if version_to_test == "base":
            # Base synonym file: label as the level name (e.g. "l1")
            version_label = cfg.command_level
        elif version_to_test is not None:
            # Numbered variant: label as "l1_v2", "l2_v3", etc.
            version_label = f"{cfg.command_level}_v{version_to_test}"
        else:
            # No command variation active: label as "default"
            version_label = "default"

        # Log version header with both original and variant instructions
        log_message("=" * 80, log_file)
        log_message(f"Testing VERSION: {version_label}", log_file)
        log_message(f"Original Command: {original_description}", log_file)
        log_message(f"Variation Command: {task_description}", log_file)
        log_message("=" * 80, log_file)

        # Per-version counters (reset for each new variant)
        task_episodes  = 0
        task_successes = 0

        # =====================================================================
        # INNER LOOP: ITERATE OVER EPISODES FOR THIS VERSION
        # =====================================================================

        for episode_idx in tqdm.tqdm(
            range(cfg.num_trials_per_task),
            desc=f"Version {version_label}",   # Progress bar label
        ):
            log_message(
                f"\n[{version_label}] Episode {episode_idx + 1}/{cfg.num_trials_per_task}",
                log_file,
            )

            # ── Initial state selection ───────────────────────────────────────
            if cfg.initial_states_path == "DEFAULT":
                # Index directly into the LIBERO-bundled states array
                initial_state = initial_states[episode_idx]

            else:
                # Custom JSON mode: look up state by task description + demo index
                # Task key: task description with spaces replaced by underscores
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"  # e.g. "demo_0", "demo_1", …

                # Skip this episode if the expert demo failed to reach the goal
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(
                        f"Skipping episode {episode_idx} (failed expert demo)",
                        log_file,
                    )
                    continue  # Advance to the next episode index without running

                # Extract the stored simulator state as a NumPy array
                initial_state = np.array(
                    all_initial_states[initial_states_task_key][episode_key]["initial_state"]
                )

            log_message(f"Starting episode {task_episodes + 1}...", log_file)

            # ── Execute the episode ───────────────────────────────────────────
            success, replay_images, replay_states = run_episode(
                cfg,
                env,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                initial_state,
                log_file,
            )

            # ── Update counters ───────────────────────────────────────────────
            task_episodes   += 1  # Episodes for this specific version
            total_episodes  += 1  # Grand total across all tasks and levels
            if success:
                task_successes  += 1
                total_successes += 1

            # ── Save rollout video ────────────────────────────────────────────
            # Saves an MP4 of the episode's third-person frames; file is named
            # using total_episodes to produce a unique, sortable identifier.
            save_rollout_video(
                {'image': replay_images, 'states': replay_states},
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                change_command=cfg.change_command,
                command_level=cfg.command_level,
                run=cfg.run_id_note,
            )

            # ── Log per-episode results ───────────────────────────────────────
            log_message(f"Success: {success}", log_file)
            log_message(f"Total episodes so far: {total_episodes}", log_file)
            log_message(
                f"Total successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)",
                log_file,
            )

        # =====================================================================
        # POST-VERSION BOOKKEEPING
        # =====================================================================

        # Compute success rate for this version; guard against zero-episode edge case
        version_success_rate = (
            float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        )

        # Log per-version summary block
        log_message(f"\n{'=' * 80}", log_file)
        log_message(f"VERSION {version_label} RESULTS:", log_file)
        log_message(f"  Episodes:     {task_episodes}", log_file)
        log_message(f"  Successes:    {task_successes}", log_file)
        log_message(f"  Success Rate: {version_success_rate:.1%}", log_file)
        log_message(f"{'=' * 80}\n", log_file)

        # Store version results for the cross-version summary printed later
        task_results_per_version[version_label] = {
            'success_rate': version_success_rate,
            'episodes':     task_episodes,
            'successes':    task_successes,
        }

        # ── Environment teardown and memory cleanup ───────────────────────────
        try:
            env.close()   # Release MuJoCo resources; some LIBERO versions omit this
        except Exception:
            pass          # Ignore AttributeError / NotImplementedError silently

        gc.collect()      # Force CPython garbage collector to reclaim MuJoCo heap memory
                          # before the next environment is instantiated

    # =========================================================================
    # CROSS-VERSION AGGREGATION AND SUMMARY
    # =========================================================================

    # Sum episodes and successes across ALL tested versions for this task
    total_task_episodes  = sum(r['episodes']  for r in task_results_per_version.values())
    total_task_successes = sum(r['successes'] for r in task_results_per_version.values())

    # Overall task success rate (safely handle the case where no episodes ran)
    task_success_rate = (
        float(total_task_successes) / float(total_task_episodes)
        if total_task_episodes > 0 else 0.0
    )

    if len(versions_to_test) > 1:
        # Print a breakdown table listing each version and its individual rate
        log_message("=" * 80, log_file)
        log_message("SUMMARY BY VERSION:", log_file)
        log_message("-" * 80, log_file)
        for version_label, results in task_results_per_version.items():
            sr   = results['success_rate']
            succ = results['successes']
            eps  = results['episodes']
            # Right-align version label in a 10-char field for visual alignment
            log_message(
                f"  {version_label:>10}: {sr:.1%} ({succ}/{eps} episodes)",
                log_file,
            )
        log_message("-" * 80, log_file)
        log_message(
            f"  {'OVERALL':>10}: {task_success_rate:.1%} "
            f"({total_task_successes}/{total_task_episodes} episodes)",
            log_file,
        )
        log_message("=" * 80, log_file)

    else:
        # Single version: print a simpler one-line summary
        log_message("=" * 80, log_file)
        log_message(
            f"TASK SUCCESS RATE: {task_success_rate:.1%} "
            f"({total_task_successes}/{total_task_episodes} episodes)",
            log_file,
        )
        log_message("=" * 80, log_file)

    # Return updated running totals + task-level summary metrics to the caller
    return (
        total_episodes,       # Updated grand total of all episodes
        total_successes,      # Updated grand total of all successes
        task_description,     # Last variant's task description (used as dict key)
        task_success_rate,    # Aggregate success rate for this task
        total_task_episodes,  # Total episodes run for this task
    )


# =============================================================================
# RESULTS TABLE PRINTER
# =============================================================================

def print_results_table(
    task_results: dict,
    command_levels: list,
    all_results: dict,
) -> None:
    """
    Print a formatted ASCII summary table of evaluation results to stdout.

    Produces two distinct table layouts depending on how many command levels
    were tested:

    Single-Level Layout
    -------------------
    Used when exactly one command level (or default) was evaluated. Shows a
    flat table with one row per task:

    .. code-block:: text

        Task                                               | Success Rate | Episodes
        ---------------------------------------------------------------------------------------------------
        pick_up_the_mug_syn_l1                            |       84.0%  |       50
        ...
        ---------------------------------------------------------------------------------------------------
        OVERALL                                            |       78.0%  |      500

    Multi-Level Layout
    ------------------
    Used when two or more command levels were evaluated (e.g. ``"all"``).
    Shows a matrix with one row per task and one column per level:

    .. code-block:: text

        Task                                     |      DEFAULT |           L1 |           L2 |           L3
        ----------------------------------------------------------------
        pick_up_the_mug...                       |       92.0%  |       84.0%  |       76.0%  |       68.0%
        ...
        ----------------------------------------------------------------
        OVERALL                                  |       88.0%  |       80.0%  |       72.0%  |       64.0%

    Both layouts are followed by a per-level summary block:

    .. code-block:: text

        SUMMARY BY COMMAND LEVEL:
        ------------------------------------------------------------
                    DEFAULT: 88.0% (440/500 episodes)
                         L1: 80.0% (400/500 episodes)
                         L2: 72.0% (360/500 episodes)
                         L3: 64.0% (320/500 episodes)
        ====================================================================================================

    Parameters
    ----------
    task_results : dict
        Nested dictionary structured as::

            {
              level_name: {
                task_description: {
                  'success_rate': float,  # e.g. 0.84
                  'episodes':     int,    # e.g. 50
                }
              }
            }

        Where ``level_name`` is one of ``"default"``, ``"l1"``, ``"l2"``,
        ``"l3"`` etc., and ``task_description`` is the natural language task
        description string used as a key.
    command_levels : list
        Ordered list of raw command level values that were iterated in
        ``eval_libero``. Each entry is either a string (``"l1"``, ``"l2"``,
        ``"l3"``) or ``None`` (which represents the default / unmodified
        instruction). The order determines the left-to-right column ordering
        in the multi-level table.
    all_results : dict
        Dictionary structured as::

            {
              level_name: {
                'success_rate':   float,
                'total_episodes': int,
                'total_successes': int,
              }
            }

        Provides overall (cross-task) aggregates for each level, used in
        the OVERALL row and the per-level summary block.

    Returns
    -------
    None
        All output is written directly to ``sys.stdout`` via ``print()``.
    """

    print("\n" + "=" * 100)
    print("DETAILED RESULTS TABLE")
    print("=" * 100)

    # ── Resolve level names — replace None with the string "default" ──────────
    # None is the internal sentinel for "no command variation"; the display
    # name is "default" for readability in printed output.
    first_level = command_levels[0]
    level_name  = first_level if first_level is not None else "default"

    # Extract the ordered list of task description strings from the first level's dict
    task_names = list(task_results[level_name].keys())

    # Build the list of display names for all tested levels
    level_names = [l if l is not None else "default" for l in command_levels]

    # =========================================================================
    # SINGLE-LEVEL TABLE LAYOUT
    # =========================================================================

    if len(level_names) == 1:

        # Column headers: left-align task name, right-align numeric columns
        print(f"{'Task':<50} | {'Success Rate':>12} | {'Episodes':>8}")
        print("-" * 100)

        for task_name in task_names:
            result = task_results[level_names[0]][task_name]
            sr  = result['success_rate']   # float in [0, 1]
            eps = result['episodes']       # int
            # :>11.1% right-aligns the percentage string in an 11-char field
            print(f"{task_name:<50} | {sr:>11.1%} | {eps:>8}")

        print("-" * 100)

        # Print the aggregate OVERALL row using pre-computed all_results values
        overall_sr  = all_results[level_names[0]]['success_rate']
        overall_eps = all_results[level_names[0]]['total_episodes']
        print(f"{'OVERALL':<50} | {overall_sr:>11.1%} | {overall_eps:>8}")

    # =========================================================================
    # MULTI-LEVEL TABLE LAYOUT
    # =========================================================================

    else:

        # Build a dynamic header string: task name + one column per level
        header = f"{'Task':<40}"
        for level_name in level_names:
            header += f" | {level_name.upper():>12}"  # Upper-case level label
        print(header)
        # Separator length: 41 chars for task name + 16 chars per level column
        print("-" * (41 + len(level_names) * 16))

        for task_name in task_names:
            row = f"{task_name:<40}"
            for level_name in level_names:
                if task_name in task_results[level_name]:
                    sr = task_results[level_name][task_name]['success_rate']
                    row += f" | {sr:>11.1%}"
                else:
                    # Task was not evaluated under this level (e.g. skipped)
                    row += f" | {'N/A':>12}"
            print(row)

        print("-" * (41 + len(level_names) * 16))

        # OVERALL row: one aggregate success rate per level
        overall_row = f"{'OVERALL':<40}"
        for level_name in level_names:
            sr = all_results[level_name]['success_rate']
            overall_row += f" | {sr:>11.1%}"
        print(overall_row)

    print("=" * 100)

    # =========================================================================
    # PER-LEVEL SUMMARY BLOCK (both layouts)
    # =========================================================================

    print("\nSUMMARY BY COMMAND LEVEL:")
    print("-" * 60)
    for level_name in level_names:
        result = all_results[level_name]
        sr    = result['success_rate']
        succ  = result['total_successes']
        total = result['total_episodes']
        # Right-align level name in a 15-char field for visual alignment
        print(f"  {level_name.upper():>15}: {sr:.1%} ({succ}/{total} episodes)")
    print("=" * 100)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """
    Top-level entry point for evaluating a trained VLA policy on LIBERO tasks.

    Decorated with ``@draccus.wrap()``, which intercepts ``sys.argv`` at
    runtime and populates a ``GenerateConfig`` instance from CLI flags and/or
    a YAML config file before calling this function. This means the function
    signature receives a fully-populated ``cfg`` object without any manual
    argument parsing.

    High-Level Execution Flow
    -------------------------
    1.  **Debug mode** (optional): If ``cfg.debug=True``, pause and wait for a
        ``debugpy`` remote debugger to attach on port 5678. This enables
        step-by-step inspection of the evaluation loop from an IDE.
    2.  **Configuration validation**: Call ``validate_config`` to catch
        misconfiguration early.
    3.  **Action chunk warning**: Check that ``cfg.num_open_loop_steps``
        matches the compile-time constant ``NUM_ACTIONS_CHUNK`` and warn if
        not — a mismatch degrades both speed and success rate.
    4.  **Seeding**: Set global random seeds for Python, NumPy, and PyTorch.
    5.  **Model loading**: Instantiate all inference components via
        ``initialize_model``.
    6.  **Image sizing**: Determine the policy input image resolution via
        ``get_image_resize_size``.
    7.  **Task suite loading**: Instantiate the LIBERO benchmark task suite
        from the registry.
    8.  **Command level enumeration**: Expand ``cfg.command_level`` into an
        ordered list of levels to iterate over (handles special values ``"all"``,
        ``"all_no_default"``, ``"default"``).
    9.  **Outer loop — command levels**: For each level:
            a. Mutate ``cfg`` to reflect the active level.
            b. Re-seed for reproducibility across levels.
            c. Open a new log file and optionally a W&B run.
            d. Determine which task IDs to evaluate (single or all).
            e. **Inner loop — tasks**: For each task, call ``run_task``
               (which itself runs all variant versions and all episodes).
            f. Accumulate per-task results into ``task_results``.
            g. Compute and log the final success rate for this level.
            h. Optionally upload metrics and the log file to W&B.
            i. Close the log file.
    10. **Print summary table**: Call ``print_results_table`` to display a
        formatted cross-task, cross-level results matrix on stdout.
    11. **Return value**: If multiple command levels were tested, return the
        arithmetic mean success rate across all levels; otherwise return the
        success rate of the single evaluated level.

    Command Level Expansion Rules
    ------------------------------
    The ``cfg.command_level`` string has several special values that control
    how the outer level loop is constructed:

    ========================  =================================================
    ``cfg.command_level``     ``command_levels`` list produced
    ========================  =================================================
    ``"all"``                 ``[None, "l1", "l2", "l3"]``  (default + all)
    ``"all_no_default"``      ``["l1", "l2", "l3"]``        (skip default)
    ``"default"``             ``[None]``                     (original only)
    ``"l1"`` / ``"l2"`` / …  ``["l1"]`` etc.               (single level)
    ``None``                  ``[None]``                     (default only)
    ========================  =================================================

    Parameters
    ----------
    cfg : GenerateConfig
        Fully-populated evaluation configuration, provided automatically by
        the ``@draccus.wrap()`` decorator from CLI arguments / YAML config.

    Returns
    -------
    float
        The evaluation success rate as a value in ``[0.0, 1.0]``:
        - If a single command level was tested: the success rate of that level.
        - If multiple levels were tested (``"all"`` or ``"all_no_default"``):
          the arithmetic mean success rate across all levels.

    Raises
    ------
    ValueError
        If ``cfg.task_id`` is specified but falls outside the valid range
        ``[0, num_tasks - 1]`` for the chosen task suite.
    AssertionError
        Propagated from ``validate_config`` for any configuration inconsistency.
    """

    # ── Step 1: Optional remote debugger ─────────────────────────────────────
    if cfg.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))   # Listen on all interfaces, port 5678
        print("Waiting for debugger attach") # Inform the user to connect their IDE
        debugpy.wait_for_client()            # Block until debugger connects

    # ── Step 2: Pre-flight configuration validation ───────────────────────────
    validate_config(cfg)

    # ── Step 3: Action chunk size consistency warning ─────────────────────────
    # NUM_ACTIONS_CHUNK is the size the model was trained with; executing fewer
    # or more actions per query can hurt both throughput and success rate.
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not "
            f"match NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK}) defined in "
            f"prismatic.vla.constants! For best performance (speed and success rate), "
            f"we recommend executing the full action chunk."
        )

    # ── Step 4: Global random seeding ─────────────────────────────────────────
    # Seeds Python random, NumPy, and PyTorch (CPU + CUDA) for reproducibility
    set_seed_everywhere(cfg.seed)

    # ── Step 5: Model and component loading ───────────────────────────────────
    model, action_head, proprio_projector, noisy_action_projector, processor = (
        initialize_model(cfg)
    )

    # ── Step 6: Policy input image dimensions ─────────────────────────────────
    resize_size = get_image_resize_size(cfg)  # Returns (H, W) tuple, e.g. (224, 224)

    # ── Step 7: LIBERO task suite instantiation ───────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()       # Registry of all suites
    task_suite     = benchmark_dict[cfg.task_suite_name]()  # Instantiate chosen suite
    num_tasks      = task_suite.n_tasks                   # Total number of tasks in suite

    # ── Step 8: Command level list expansion ──────────────────────────────────
    # Expand the single command_level string into an ordered list of levels to loop over

    if cfg.change_command and cfg.command_level == "all":
        # Test original instruction + all three linguistic variation levels
        command_levels = [None, "l1", "l2", "l3"]
        log_message_prefix = "Testing all command levels (default, l1, l2, l3)"

    elif cfg.change_command and cfg.command_level == "all_no_default":
        # Test only the three variation levels, skip the original instruction
        command_levels = ["l1", "l2", "l3"]
        log_message_prefix = "Testing command levels (l1, l2, l3) without default"

    elif cfg.change_command and cfg.command_level == "default":
        # Force-test only the original (un-modified) instruction
        command_levels = [None]
        log_message_prefix = "Testing default command only"

    elif cfg.change_command and cfg.command_level is not None:
        # Test a single explicitly-named variation level
        command_levels = [cfg.command_level]
        log_message_prefix = f"Testing command level: {cfg.command_level}"

    else:
        # command variation disabled or command_level is None: default only
        command_levels = [None]
        log_message_prefix = "Testing with default commands"

    # ── Accumulators for cross-level results ──────────────────────────────────
    all_results  = {}   # {level_name: {'success_rate', 'total_episodes', 'total_successes'}}
    task_results = {}   # {level_name: {task_description: {'success_rate', 'episodes'}}}

    # =========================================================================
    # OUTER LOOP: ITERATE OVER COMMAND LEVELS
    # =========================================================================

    for level in command_levels:

        # ── Resolve display name: None → "default" ────────────────────────────
        current_level_name = level if level is not None else "default"

        # Mutate cfg to reflect the active level for this iteration.
        # change_command is set to False when level is None (default mode).
        cfg.command_level = level
        cfg.change_command = (level is not None)

        # Re-seed before each level so all levels start from the same random state,
        # ensuring that stochastic differences between levels are meaningful and not
        # just artefacts of different RNG trajectories.
        set_seed_everywhere(cfg.seed)

        # Open a new log file for this level (each level gets its own .txt file)
        log_file, local_log_filepath, run_id = setup_logging(cfg)

        # Log level header
        log_message("=" * 80, log_file)
        log_message(f"EVALUATING: {current_level_name.upper()}", log_file)
        log_message("=" * 80, log_file)
        log_message(f"Task suite: {cfg.task_suite_name}", log_file)
        log_message(log_message_prefix, log_file)

        # Initialise nested results dict for this level
        task_results[current_level_name] = {}

        # Reset per-level episode counters
        total_episodes, total_successes = 0, 0

        # ── Determine which task IDs to evaluate ──────────────────────────────
        if cfg.task_id is not None:
            # Single-task mode: validate the requested ID first
            if cfg.task_id < 0 or cfg.task_id >= num_tasks:
                raise ValueError(
                    f"task_id {cfg.task_id} out of range [0, {num_tasks - 1}]"
                )
            task_ids = [cfg.task_id]
            log_message(f"Evaluating only task {cfg.task_id}", log_file)
        else:
            # Full-suite mode: evaluate every task in sequential order
            task_ids = range(num_tasks)

        # =====================================================================
        # INNER LOOP: ITERATE OVER TASKS
        # =====================================================================

        for task_id in tqdm.tqdm(task_ids, desc=f"Level {current_level_name}"):

            # run_task returns updated running totals + per-task summary metrics
            (
                total_episodes,
                total_successes,
                task_name,      # Task description string used as dict key
                task_sr,        # Aggregate success rate for this task
                task_eps,       # Total episodes run for this task
            ) = run_task(
                cfg,
                task_suite,
                task_id,
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

            # Store per-task results under the current level key
            task_results[current_level_name][task_name] = {
                'success_rate': task_sr,
                'episodes':     task_eps,
            }

        # ── Compute final success rate for this level ─────────────────────────
        final_success_rate = (
            float(total_successes) / float(total_episodes)
            if total_episodes > 0 else 0.0
        )

        # Store level-level aggregates for the summary table
        all_results[current_level_name] = {
            'success_rate':    final_success_rate,
            'total_episodes':  total_episodes,
            'total_successes': total_successes,
        }

        # ── Log final results for this level ──────────────────────────────────
        log_message("=" * 80, log_file)
        log_message(f"RESULTS FOR {current_level_name.upper()}:", log_file)
        log_message("=" * 80, log_file)
        log_message(f"Total episodes: {total_episodes}", log_file)
        log_message(f"Total successes: {total_successes}", log_file)
        log_message(
            f"Overall success rate: {final_success_rate:.4f} "
            f"({final_success_rate * 100:.1f}%)",
            log_file,
        )
        log_message("=" * 80, log_file)

        # ── Optional W&B logging ──────────────────────────────────────────────
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{current_level_name}":  final_success_rate,
                f"num_episodes/{current_level_name}":  total_episodes,
            })
            wandb.save(local_log_filepath)  # Upload the .txt log as a W&B artefact

        # ── Close the log file for this level ────────────────────────────────
        if log_file:
            log_file.close()  # Flush OS buffers and release the file descriptor

    # =========================================================================
    # FINAL SUMMARY TABLE AND RETURN VALUE
    # =========================================================================

    # Print the formatted ASCII results table to stdout
    print_results_table(task_results, command_levels, all_results)

    # Return value: mean over all levels if multiple were tested, otherwise
    # the single final_success_rate from the last loop iteration.
    if len(command_levels) > 1:
        avg_success_rate = (
            sum(r['success_rate'] for r in all_results.values()) / len(all_results)
        )
        return avg_success_rate   # Scalar mean in [0.0, 1.0]
    else:
        return final_success_rate  # Single-level rate in [0.0, 1.0]


# =============================================================================
# SCRIPT ENTRY GUARD
# =============================================================================

if __name__ == "__main__":
    # This block executes only when the script is run directly, e.g.:
    #   python run_libero_eval.py --pretrained_checkpoint /path/to/ckpt ...
    #
    # The @draccus.wrap() decorator on eval_libero handles all CLI argument
    # parsing automatically, so no manual argparse setup is required here.
    # When the script is imported as a module, this block is skipped entirely.
    eval_libero()