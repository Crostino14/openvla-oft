"""
run_libero_ablation.py
======================

Ablation study for probing *keyword-shortcut* reliance in Vision-Language-Action
(VLA) policies evaluated on LIBERO-Goal tasks.

Scientific Motivation
---------------------
Imitation-learning policies conditioned on natural-language instructions can
develop superficial shortcuts: rather than grounding the full sentence meaning,
a model may learn to associate a single dominant keyword (e.g. ``"stove"``) with
a specific motor sequence and ignore the remaining words. This behaviour is not
detectable by standard evaluations — the policy still succeeds with the full
instruction — but it indicates fragile language grounding that breaks under
paraphrasing, synonym substitution, or adversarial rephrasing.

This script operationalises the keyword-shortcut hypothesis as a controlled
ablation study [Liu et al., NeurIPS 2024; LIBERO-Para, 2026].
For a selected LIBERO-Goal task it:

1. Defines a set of **keyword-only BDDL variants**, each containing a
   deliberately truncated or rearranged instruction (e.g. ``"stove"`` alone,
   ``"bowl stove"``, ``"Turn on"`` without the object noun).
2. Runs a full ``cfg.num_trials_per_task`` (default 50) episode rollout of
   the trained VLA policy under each variant, using the **same initial
   simulator states** as the original task to ensure fair comparison.
3. Logs per-variant success rates to a ``.txt`` file and optionally to
   Weights & Biases.
4. Prints a formatted ASCII summary table for immediate inspection.

A policy that achieves the same success rate under ``"stove"`` as under
``"Turn on the stove"`` is very likely exploiting a keyword shortcut: it
executes the correct motor sequence based on a single object noun, not
from understanding the full instruction.

Scope and Constraints
---------------------
- Restricted to ``libero_goal`` suite (enforced by ``validate_config``).
- Currently supports task IDs 7 and 8 (1-indexed: Task 8 "Turn on the stove"
  and Task 9 "Put the bowl on the plate"). Adding a new task requires:
  (1) authoring custom BDDL files with truncated instructions, and
  (2) extending ``ABLATION_CONFIGS`` inside ``get_ablation_tasks``.
- Does **not** compute statistical significance; with N=50 episodes the ±95%
  confidence interval on a binary success rate is approximately ±7%, so
  differences < ~10 percentage points should be interpreted cautiously.

Typical CLI Usage
-----------------
    python run_libero_ablation.py \\
        --pretrained_checkpoint /path/to/checkpoint \\
        --ablation_task_id 7 \\
        --num_trials_per_task 50 \\
        --seed 42 \\
        --run_id_note keyword_shortcut_v1

References
----------
- Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot
  Learning", NeurIPS 2024. https://github.com/Lifelong-Robot-Learning/LIBERO
- Kim et al., "OpenVLA-OFT: Fine-Tuning Vision-Language-Action Models",
  2025. https://openvla-oft.github.io
- LIBERO-Para (2026): introduces PRIDE metric for VLA paraphrase robustness.
  https://arxiv.org/abs/2603.28301
- LIBERO-PRO (2025): shows VLA failure under task-instruction perturbation.
  https://arxiv.org/abs/2510.03827

Author: Agostino Cardamone
"""


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import json        # JSON deserialisation for custom initial states files
import logging     # Levelled structured logging to stdout
import os          # Filesystem utilities: os.makedirs, os.path.join
import sys         # sys.path manipulation to expose local packages

from collections import deque       # Fixed-capacity FIFO queue for action chunking
from dataclasses import dataclass   # @dataclass decorator for declarative config struct
from enum import Enum               # Enum base class for strongly-typed task suite names
from pathlib import Path            # Object-oriented, OS-independent file path handling
from typing import Optional, Union  # PEP 484 generic type annotations


# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

import draccus       # Parses Python dataclass fields as CLI flags and/or YAML config
import numpy as np   # Numerical arrays: state concatenation, initial state casting
import tqdm          # Progress bar decorator wrapping episode/task iterables

from libero.libero import benchmark  # LIBERO benchmark registry; provides task suites
                                     # via ``benchmark.get_benchmark_dict()``

import wandb  # Weights & Biases experiment tracking.
              # NOTE: imported unconditionally at module level. Unlike the other
              # evaluation scripts that use lazy ``import wandb`` inside the
              # ``if cfg.use_wandb`` blocks, this script always requires wandb
              # to be installed in the Python environment.


# =============================================================================
# PROJECT ROOT RESOLUTION
# =============================================================================
# This block ensures that the top-level packages of the OpenVLA-OFT repository
# (experiments/, prismatic/) are importable regardless of the current working
# directory from which the script is launched (e.g., SLURM job directory,
# interactive session, or IDE run configuration).

from pathlib import Path  # Re-imported explicitly here for clarity in this block

current_file = Path(__file__).resolve()            # Absolute, symlink-resolved path to
                                                    # this .py file
project_root = current_file.parent.parent.parent   # Navigate three directories up:
                                                    # run_libero_ablation.py
                                                    #   → experiments/libero/
                                                    #     → experiments/
                                                    #       → openvla-oft/   ← project_root
sys.path.insert(0, str(project_root))              # Prepend the repo root so local
                                                    # packages shadow any installed
                                                    # versions with the same name


# =============================================================================
# INTERNAL UTILITIES — LIBERO-SPECIFIC HELPERS
# =============================================================================

from experiments.libero.utils.libero_utils import (
    get_libero_dummy_action,  # Returns a neutral zero-velocity action for warm-up
    get_libero_env,           # Environment factory; supports custom BDDL override via
                              # ``ablation_bddl_file`` keyword argument
    get_libero_image,         # Extracts the third-person RGB frame from an obs dict
    get_libero_wrist_image,   # Extracts the wrist camera RGB frame from an obs dict
    quat2axisangle,           # Converts quaternion (qx,qy,qz,qw) to axis-angle (ax,ay,az)
    save_rollout_video,       # Renders a list of RGB frames to an MP4 replay video
)


# =============================================================================
# INTERNAL UTILITIES — OPENVLA MODEL COMPONENTS
# =============================================================================

from experiments.openvla_utils import (
    get_action_head,            # Loads the continuous action prediction head
                                # (L1 regression or DDIM diffusion variant)
    get_noisy_action_projector, # Loads the noise-step conditioning MLP for DDIM
    get_processor,              # Returns the VLA tokenizer + image pre-processor
    get_proprio_projector,      # Loads the MLP mapping 8-dim proprio → LLM embed dim
    resize_image_for_policy,    # Resizes a raw NumPy RGB image to (H, W) policy input
)


# =============================================================================
# INTERNAL UTILITIES — GENERAL ROBOT EXPERIMENT HELPERS
# =============================================================================

from experiments.robot_utils import (
    DATE_TIME,                # ISO-8601 timestamp string captured at module import time;
                              # used to construct unique run IDs
    get_image_resize_size,    # Derives the policy's expected (H, W) input dimensions
    invert_gripper_action,    # Flips gripper sign to undo OpenVLA's training convention
    normalize_gripper_action, # Maps raw gripper output to binary {−1, +1} for LIBERO
    set_seed_everywhere,      # Sets Python, NumPy, and PyTorch random seeds for
                              # fully deterministic evaluation
)


# =============================================================================
# VLA INFERENCE ENTRY POINTS
# =============================================================================
# Second import block from openvla_utils for the two inference-time functions.
# These are separated from the model-component imports above for clarity.

from experiments.openvla_utils import (
    get_vla,        # Instantiates the full VLA model from a checkpoint
    get_vla_action, # Runs a VLA forward pass; returns an action chunk (list of vectors)
)

from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # Expected number of actions per
                                                        # policy query (e.g. 8); used to
                                                        # validate cfg.num_open_loop_steps


# =============================================================================
# MODULE-LEVEL LOGGER
# =============================================================================

# Configure the root logger: ISO timestamp prefix, severity level, message body.
# StreamHandler directs all output to sys.stdout (console).
logging.basicConfig(
    level=logging.INFO,                                   # Show INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",     # e.g. "2026-04-16 11:00:00,123 [INFO] ..."
    handlers=[logging.StreamHandler()],                   # Write to stdout
)
logger = logging.getLogger(__name__)  # Child logger namespaced to this module
                                      # (resolves to "run_libero_ablation")


# =============================================================================
# TASK SUITE ENUMERATION
# =============================================================================

class TaskSuite(str, Enum):
    """
    Enumeration of task suites supported by this ablation script.

    Inherits from both ``str`` and ``Enum`` so that instances compare equal
    to their underlying string value. This is required because LIBERO's
    benchmark registry uses plain string keys (e.g. ``"libero_goal"``), and
    ``cfg.task_suite_name`` must match those keys exactly in equality checks
    and dict lookups.

    Currently only ``LIBERO_GOAL`` is supported. All ablation BDDL files and
    initial state files are authored for the libero_goal scene, which features
    a single tabletop layout with 10 manipulation tasks. Extending the ablation
    to other suites would require new BDDL authoring and is not implemented.

    Members
    -------
    LIBERO_GOAL : str
        Value ``"libero_goal"``. The only valid choice for
        ``cfg.task_suite_name`` in this script (enforced by ``validate_config``).
    """
    LIBERO_GOAL = "libero_goal"


# =============================================================================
# STEP BUDGET PER TASK SUITE
# =============================================================================

TASK_MAX_STEPS = {
    TaskSuite.LIBERO_GOAL: 300,  # 300 active control steps × 20 Hz = 15 seconds per episode
}
"""
dict[TaskSuite, int]
    Maps each supported task suite to the maximum number of **active**
    simulator steps (i.e., steps after the warm-up phase) allowed per episode.

    A hard step budget is used instead of a heuristic early-stopping rule so
    that all ablation variants receive exactly the same opportunity to succeed.
    This makes success rate differences across variants attributable solely
    to the linguistic content of the instruction, not to step-budget asymmetry.

    The value 300 for LIBERO-Goal corresponds to 15 seconds of manipulation
    time at 20 Hz control frequency, which is sufficient for all single-step
    manipulation tasks in that suite (pick-place, push, open, turn-on).
"""


# =============================================================================
# ABLATION TASK REGISTRY
# =============================================================================

def get_ablation_tasks(task_id: int) -> dict:
    """
    Return the ablation configuration for a specific LIBERO-Goal task.

    Maintains an internal ``ABLATION_CONFIGS`` dictionary that maps
    0-indexed task IDs to structured test definitions. Each test definition
    pairs a **custom BDDL filename** (which embeds a truncated instruction
    in its ``:language`` field) with an ``expected_command`` string describing
    the keyword content of that instruction.

    Ablation Test Design Rationale
    --------------------------------
    The tests are designed to systematically remove semantic components from
    the original natural-language instruction and measure the policy's
    success rate under each degraded form:

    **Task 7 — "Turn on the stove"** (4 tests)

    +----------+------------------+-----------------------------------------------+
    | Key      | expected_command | What this tests                               |
    +==========+==================+===============================================+
    | stove1   | ``"stove"``      | Single object noun: minimal possible cue      |
    +----------+------------------+-----------------------------------------------+
    | stove2   | ``"bowl stove"`` | Distractor noun + target: selectivity check   |
    +----------+------------------+-----------------------------------------------+
    | stove3   | ``"plate stove"``| Different distractor + target                 |
    +----------+------------------+-----------------------------------------------+
    | stove4   | ``"Turn on"``    | Verb phrase only: no object noun present      |
    +----------+------------------+-----------------------------------------------+

    If the policy succeeds equally well with ``"stove"`` alone as with the
    full instruction, it relies on an object-keyword shortcut. If it fails
    with ``"Turn on"`` but succeeds with ``"stove"``, it ignores the action
    verb and acts purely on the object name.

    **Task 8 — "Put the bowl on the plate"** (5 tests)

    +---------------+------------------+------------------------------------------+
    | Key           | expected_command | What this tests                          |
    +===============+==================+==========================================+
    | bowl_plate1   | ``"bowl"``       | Grasped object only                      |
    +---------------+------------------+------------------------------------------+
    | bowl_plate2   | ``"plate"``      | Target object only                       |
    +---------------+------------------+------------------------------------------+
    | bowl_plate3   | ``"bowl plate"`` | Both objects, canonical order, no verb   |
    +---------------+------------------+------------------------------------------+
    | bowl_plate4   | ``"Put plate"``  | Action verb + target only                |
    +---------------+------------------+------------------------------------------+
    | bowl_plate5   | ``"plate bowl"`` | Both objects, **reversed** order         |
    +---------------+------------------+------------------------------------------+

    The extra ``bowl_plate5`` test (not present for task 7) probes **word-order
    sensitivity**: if the policy treats ``"bowl plate"`` and ``"plate bowl"``
    identically, it is not tracking which object is the source and which is
    the target — a critical failure mode for pick-and-place tasks.

    Extending the Registry
    ----------------------
    To add ablation support for a new task:
    1. Author one BDDL file per test variant in the libero_goal BDDL directory,
       embedding the desired truncated string in ``:language "..."``.
    2. Add the new task ID as a key in ``ABLATION_CONFIGS`` with its
       ``"task_name"`` and ``"tests"`` dict.
    3. No other code changes are needed — ``eval_ablation`` iterates over
       whatever keys ``get_ablation_tasks`` returns.

    Parameters
    ----------
    task_id : int
        0-indexed LIBERO-Goal task identifier. Corresponds to the task's
        position in the suite (Task N in the UI → ``task_id = N-1``).
        Currently supported: ``7`` (Task 8) and ``8`` (Task 9).

    Returns
    -------
    config : dict
        Dictionary with two keys:
        - ``"task_name"`` : str — human-readable task description.
        - ``"tests"`` : dict[str, dict] — mapping from test key to:
            - ``"bddl_file"`` : str — filename of the custom ablation BDDL
              (relative to the libero_goal BDDL directory).
            - ``"expected_command"`` : str — the keyword-only instruction
              string embedded in that BDDL's ``:language`` field.

    Raises
    ------
    ValueError
        If ``task_id`` is not present in ``ABLATION_CONFIGS``.
        The error message lists the full set of available IDs to aid debugging.

    Examples
    --------
    >>> cfg = get_ablation_tasks(7)
    >>> cfg["task_name"]
    'Turn on the stove'
    >>> list(cfg["tests"].keys())
    ['stove1', 'stove2', 'stove3', 'stove4']
    """
    # ── Internal registry (defined locally to keep it encapsulated) ───────────
    ABLATION_CONFIGS = {

        # ── Task 8: "Turn on the stove" (0-indexed: task_id = 7) ─────────────
        7: {
            "task_name": "Turn on the stove",
            "tests": {
                "stove1": {
                    "bddl_file": "turn_on_the_stove_ablation_stove1.bddl",
                    "expected_command": "stove",        # Object name only
                },
                "stove2": {
                    "bddl_file": "turn_on_the_stove_ablation_stove2.bddl",
                    "expected_command": "bowl stove",   # Distractor + target
                },
                "stove3": {
                    "bddl_file": "turn_on_the_stove_ablation_stove3.bddl",
                    "expected_command": "plate stove",  # Different distractor + target
                },
                "stove4": {
                    "bddl_file": "turn_on_the_stove_ablation_stove4.bddl",
                    "expected_command": "Turn on",      # Action verb only, no object
                },
            },
        },

        # ── Task 9: "Put the bowl on the plate" (0-indexed: task_id = 8) ─────
        8: {
            "task_name": "Put the bowl on the plate",
            "tests": {
                "bowl_plate1": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate1.bddl",
                    "expected_command": "bowl",         # Source object only
                },
                "bowl_plate2": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate2.bddl",
                    "expected_command": "plate",        # Target object only
                },
                "bowl_plate3": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate3.bddl",
                    "expected_command": "bowl plate",   # Both objects, canonical order
                },
                "bowl_plate4": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate4.bddl",
                    "expected_command": "Put plate",    # Verb + target only
                },
                "bowl_plate5": {
                    "bddl_file": "put_the_bowl_on_the_plate_ablation_bowl_plate5.bddl",
                    "expected_command": "plate bowl",   # Reversed object order
                },
            },
        },
    }

    # Guard: fail fast with a descriptive error if the task_id is unknown
    if task_id not in ABLATION_CONFIGS:
        raise ValueError(
            f"No ablation configuration found for task_id={task_id}. "
            f"Available task IDs: {list(ABLATION_CONFIGS.keys())}"
        )

    return ABLATION_CONFIGS[task_id]  # Return the matching configuration dict


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class GenerateConfig:
    # fmt: off
    
    #################################################################################################################
    # Ablation-specific parameters
    #################################################################################################################
    ablation_task_id: int = 7  # Task ID to run ablation on (0-indexed, default=7 for Task 8)
    
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
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_GOAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
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
    # fmt: on


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(cfg: GenerateConfig) -> None:
    """
    Validate evaluation configuration before any heavyweight resource is loaded.

    Performs five sequential checks that fail fast with descriptive messages,
    preventing misleading errors deep in model loading or environment creation.

    Checks (in order)
    -----------------
    1. ``pretrained_checkpoint`` is not ``None``. An empty string is still
       allowed here (it will fail later in ``get_vla`` with a clearer error);
       ``None`` would cause silent failures in ``str()`` comparisons.
    2. If the checkpoint path contains the substring ``"image_aug"``, then
       ``center_crop`` must be ``True``. Checkpoints trained with random-crop
       augmentation expect center-cropped inputs at eval time; without this,
       the effective crop window differs from the training distribution.
    3. ``load_in_8bit`` and ``load_in_4bit`` are mutually exclusive. The
       bitsandbytes backend cannot apply both quantisation schemes
       simultaneously.
    4. ``task_suite_name`` must equal ``"libero_goal"`` (i.e.
       ``TaskSuite.LIBERO_GOAL``). All ablation BDDL files and initial states
       are authored for this specific scene.
    5. ``ablation_task_id`` must have an entry in ``ABLATION_CONFIGS``.
       Verified by calling ``get_ablation_tasks`` and re-raising its
       ``ValueError`` with a prefixed context message.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration object populated by draccus from CLI / YAML.

    Raises
    ------
    AssertionError
        If any of checks 1–4 fail.
    ValueError
        If check 5 fails (unknown ``ablation_task_id``).
    """
    # Check 1: checkpoint must not be None
    assert cfg.pretrained_checkpoint is not None, \
        "pretrained_checkpoint must not be None!"

    # Check 2: image-augmented checkpoints require center crop at eval time
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, (
            "Expecting `center_crop==True` because model was trained with "
            "image augmentations!"
        )

    # Check 3: quantisation modes are mutually exclusive
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), \
        "Cannot use both 8-bit and 4-bit quantization!"

    # Check 4: only libero_goal is supported for keyword-shortcut ablation
    assert cfg.task_suite_name == TaskSuite.LIBERO_GOAL, \
        "Ablation only works with libero_goal!"

    # Check 5: ablation_task_id must correspond to a registered config entry
    try:
        get_ablation_tasks(cfg.ablation_task_id)
    except ValueError as e:
        # Wrap with context prefix so the caller knows which check failed
        raise ValueError(f"Invalid ablation_task_id: {e}")


# =============================================================================
# MODEL INITIALISATION
# =============================================================================

def initialize_model(cfg: GenerateConfig):
    """
    Instantiate the VLA backbone and all optional neural network components.

    Loads each module conditionally based on architecture flags in ``cfg``,
    minimising GPU memory usage when optional heads or projectors are disabled.
    CUDA memory is cleared before weight loading to reduce fragmentation from
    prior operations.

    Component Loading Order
    -----------------------
    1. **VLA backbone** — always loaded via ``get_vla(cfg)``.
    2. **Proprio projector** (conditional on ``use_proprio``) — MLP mapping
       the 8-dim proprioceptive state into the LLM embedding space.
    3. **Action head** (conditional on ``use_l1_regression`` or
       ``use_diffusion``) — continuous action prediction head. Includes a
       graceful fallback: if the weight file is not found, it is assumed that
       the head is fused into the LoRA checkpoint and ``action_head`` is set
       to ``None``. A warning is logged (in Italian, as legacy development
       language).
    4. **Noisy action projector** (conditional on ``use_diffusion``) — DDIM
       noise-step conditioning MLP. Only loaded when diffusion is active.
    5. **Processor** (conditional on ``model_family == "openvla"``) — VLA
       tokenizer and image pre-processor. After loading, ``check_unnorm_key``
       is called to resolve and validate the action un-normalisation key.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration specifying the checkpoint path, architecture flags,
        and quantisation settings.

    Returns
    -------
    model : torch.nn.Module
        Loaded VLA backbone in evaluation mode (``model.eval()``).
    action_head : torch.nn.Module or None
        Continuous action head; ``None`` if not used or fused into the model.
    proprio_projector : torch.nn.Module or None
        Proprioceptive state projector; ``None`` if ``use_proprio=False``.
    noisy_action_projector : torch.nn.Module or None
        DDIM noise projector; ``None`` if ``use_diffusion=False``.
    processor : transformers.ProcessorMixin or None
        VLA tokenizer/image processor; ``None`` if not OpenVLA.
    """
    import torch  # Imported inside function to keep the module importable on
                  # systems without CUDA (e.g., for unit testing)

    # Free any cached-but-unused GPU memory before loading large model weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    # Release all cached allocations
        torch.cuda.synchronize()    # Wait for all pending CUDA kernels to finish

    # ── 1. VLA backbone (always loaded) ──────────────────────────────────────
    model = get_vla(cfg)  # Returns the full VLA model placed on GPU(s)

    # ── 2. Proprioceptive state projector ─────────────────────────────────────
    proprio_projector = None  # Default: not used
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,   # Must match the LLM hidden dimension (e.g. 4096 for 7B)
            proprio_dim=8,   # State vector length: 3 (XYZ) + 3 (axis-angle) + 2 (gripper)
        )

    # ── 3. Continuous action head ─────────────────────────────────────────────
    action_head = None  # Default: not used
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            logger.info("Action head caricato separatamente")   # "Loaded separately"
        except (AssertionError, FileNotFoundError) as e:
            # Weight file not found → assume the head is integrated into the
            # LoRA checkpoint. This is a valid configuration for some training runs.
            logger.warning("Action head non trovato come file separato")
            logger.warning("Assumo sia integrato nel modello (checkpoint LoRA)")
            action_head = None  # Inference will use the fused head from the backbone

    # ── 4. DDIM noise-step projector ──────────────────────────────────────────
    noisy_action_projector = None  # Default: not used
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # ── 5. Input processor (OpenVLA only) ─────────────────────────────────────
    processor = None  # Default: not used (non-OpenVLA models)
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)   # Tokenizer + image pre-processor
        check_unnorm_key(cfg, model)     # Resolve action un-norm key in-place

    return model, action_head, proprio_projector, noisy_action_projector, processor


# =============================================================================
# ACTION UN-NORMALISATION KEY RESOLUTION
# =============================================================================

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """
    Resolve and validate the action un-normalisation key in the VLA model.

    VLA models store per-dimension action statistics (mean and standard
    deviation) in ``model.norm_stats``, a dict keyed by the dataset name
    used during training (e.g. ``"libero_goal"`` or ``"libero_goal_no_noops"``).
    These statistics are needed at inference time to de-standardise the
    model's raw predicted actions back to the original action space.

    This function finds the correct key and writes it back into ``cfg.unnorm_key``
    so that ``get_vla_action`` can use it without further configuration.

    Resolution Priority
    -------------------
    1. If ``cfg.unnorm_key`` is non-empty and exists in ``model.norm_stats``,
       use it as-is (user explicitly provided a valid key).
    2. Otherwise, start from ``cfg.task_suite_name`` (e.g. ``"libero_goal"``).
    3. If the plain key is absent, try ``"{key}_no_noops"`` (dataset variant
       that filtered out no-operation steps during data collection).
    4. If neither is found, raise ``AssertionError``.

    Note: Unlike ``run_libero_eval_task_comp.check_unnorm_key``, this function
    does **not** try a ``"_noops"`` variant as a third fallback. This is
    intentional, as the ablation study is expected to always use a checkpoint
    trained on the standard ``libero_goal`` or ``libero_goal_no_noops`` splits.

    Side Effects
    ------------
    Mutates ``cfg.unnorm_key`` in place with the resolved key string.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration object whose ``unnorm_key`` field is updated.
    model : torch.nn.Module
        Loaded VLA model with a ``norm_stats`` dict attribute.

    Raises
    ------
    AssertionError
        If no valid key can be resolved from the model's ``norm_stats``.
    """
    # Log available keys for debugging mismatches in the log file
    logger.info(f"Available norm_stats keys: {list(model.norm_stats.keys())}")
    logger.info(f"cfg.unnorm_key from config: '{cfg.unnorm_key}'")

    # Priority 1: use user-specified key if it exists
    if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
        logger.info(f"Using user-specified unnorm_key: {cfg.unnorm_key}")
        return  # Key is valid; nothing to change

    # Priority 2: derive from the task suite name
    unnorm_key = cfg.task_suite_name  # e.g. "libero_goal"

    # Priority 3: try the "_no_noops" dataset variant
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"  # e.g. "libero_goal_no_noops"

    # Fail explicitly if no valid key was found; guides the user to check
    # which dataset key was used during training
    assert unnorm_key in model.norm_stats, (
        f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    )

    cfg.unnorm_key = unnorm_key  # Persist resolved key; used by get_vla_action


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(cfg: GenerateConfig, task_name: str):
    """
    Initialise per-run file logging and optionally a Weights & Biases run.

    Constructs a unique, human-readable run ID that encodes the task number,
    task name, model family, ISO timestamp, and an optional user annotation.
    This ID is used both as the ``.txt`` log filename and as the W&B run name,
    making it easy to correlate logs across different storage systems.

    Run ID Format
    -------------
    ``ABLATION-Task{N}-{safe_task_name}-{model_family}-{DATE_TIME}[--{run_id_note}]``

    where:
    - ``N`` = ``cfg.ablation_task_id + 1`` (1-indexed, matches the UI label)
    - ``safe_task_name`` = task name lowercased with spaces replaced by ``_``
    - ``DATE_TIME`` = ISO-8601 timestamp captured at module import time

    Example (task 8, no note):
        ``ABLATION-Task8-turn_on_the_stove-openvla-20260416_140000``

    Example (with run note):
        ``ABLATION-Task8-turn_on_the_stove-openvla-20260416_140000--v2_ablation``

    The log directory is created with ``exist_ok=True`` so repeated runs do
    not fail if the directory already exists from a previous experiment.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration providing task ID, model family, log directory path,
        W&B credentials, and optional run note.
    task_name : str
        Human-readable task name (e.g. ``"Turn on the stove"``). Used directly
        in the run ID after sanitisation (lowercase + underscore substitution).

    Returns
    -------
    log_file : io.TextIOWrapper
        Open file handle (write mode) to the ``.txt`` log file. The caller
        is responsible for closing it at the end of the evaluation run.
    local_log_filepath : str
        Absolute path to the opened ``.txt`` log file.
    run_id : str
        The unique run identifier used for the log filename and W&B run name.
    """
    # Sanitise task name for use as a filename component:
    # "Turn on the stove" → "turn_on_the_stove"
    safe_task_name = task_name.replace(" ", "_").lower()

    # Build unique run ID: 1-indexed task number + sanitised name + model + timestamp
    run_id = (
        f"ABLATION-Task{cfg.ablation_task_id + 1}-"
        f"{safe_task_name}-"
        f"{cfg.model_family}-"
        f"{DATE_TIME}"
    )
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"  # Append user annotation if provided

    # Create log directory (including any missing parent directories)
    os.makedirs(cfg.local_log_dir, exist_ok=True)

    # Build absolute path and open the log file in write mode
    # (truncates any existing file with the same name)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Optionally initialise a Weights & Biases run for real-time metric streaming
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,   # W&B team or user namespace
            project=cfg.wandb_project, # W&B project board to receive metrics
            name=run_id,               # Run name visible in the W&B UI
        )

    return log_file, local_log_filepath, run_id


# =============================================================================
# DUAL-SINK LOG HELPER
# =============================================================================

def log_message(message: str, log_file=None) -> None:
    """
    Write a message to the module-level console logger and optionally to a file.

    Every evaluation status update, episode result, and section separator
    passes through this function to ensure all output appears in both the
    terminal (via the Python logging system) and the persistent ``.txt`` log
    file simultaneously.

    The file is flushed after every write so that log data is persisted to
    disk immediately, even if the process terminates abruptly (e.g. due to a
    SLURM job timeout or GPU OOM).

    Parameters
    ----------
    message : str
        The text to log. A newline character is appended when writing to the
        file; the logger handler adds its own newline when writing to stdout.
    log_file : io.TextIOWrapper or None, optional
        Open, writable file handle. When ``None``, the message is emitted
        only to the console logger.

    Returns
    -------
    None
    """
    logger.info(message)       # Always emit to console at INFO level
    if log_file:
        log_file.write(message + "\n")  # Write with explicit newline delimiter
        log_file.flush()                # Force OS write-through; prevents data loss


# =============================================================================
# INITIAL STATE LOADER
# =============================================================================

def load_initial_states(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    log_file=None,
):
    """
    Load initial simulator states for a specific LIBERO-Goal task.

    Supports two modes of operation, selected by ``cfg.initial_states_path``:

    Default Mode (``"DEFAULT"``)
    ----------------------------
    Uses the LIBERO-bundled states returned by
    ``task_suite.get_task_init_states(task_id)``. These are the same states
    used in the standard LIBERO evaluation benchmark. Using them ensures that
    ablation results are directly comparable to baseline results produced by
    the standard evaluation script.

    Custom Mode (any other string)
    ------------------------------
    Loads a custom JSON file from ``cfg.initial_states_path``. The JSON is
    expected to be a nested dict with structure:
    ``{task_description_with_underscores: {demo_N: {initial_state: [...], success: bool}}}``.
    This mode enables filtering: episodes where the expert demonstration
    failed (``"success": false``) can be skipped in the caller (see
    ``run_ablation_task``).

    In both modes the LIBERO default ``initial_states`` array is always loaded
    (it is needed by the ablation task runner even in custom mode, e.g. for
    episode ordering). The custom ``all_initial_states`` dict is returned as
    ``None`` in default mode.

    Parameters
    ----------
    cfg : GenerateConfig
        Configuration providing ``initial_states_path``.
    task_suite : libero.libero.benchmark.TaskSuite
        Instantiated LIBERO task suite (e.g. the libero_goal suite).
    task_id : int
        0-indexed task ID within the suite.
    log_file : io.TextIOWrapper or None, optional
        Open log file for status messages.

    Returns
    -------
    initial_states : sequence
        LIBERO-bundled initial states for ``task_id``. Indexed as
        ``initial_states[episode_idx]`` in the episode loop.
    all_initial_states : dict or None
        Custom JSON initial states dict (loaded from file) when
        ``cfg.initial_states_path != "DEFAULT"``; ``None`` otherwise.
    """
    # Always load the LIBERO default states — needed in both modes
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        # Custom mode: load a JSON file of pre-recorded / filtered states
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)  # Deserialise full nested JSON dict
        log_message(
            f"Using initial states from {cfg.initial_states_path}",
            log_file,
        )
        return initial_states, all_initial_states  # Both arrays available to caller

    else:
        # Default mode: use LIBERO-bundled states only
        log_message("Using default initial states", log_file)
        return initial_states, None  # all_initial_states not needed


def prepare_observation(obs: dict, resize_size: tuple) -> tuple:
    """
    Convert a raw LIBERO observation dictionary into the structured format
    expected by the VLA model for a single inference call.

    The VLA's ``get_vla_action`` function requires a specific observation
    dictionary with exactly three keys: two resized RGB camera frames and
    one proprioceptive state vector. This function performs all necessary
    extraction, transformation, and concatenation steps to produce that
    dictionary from the raw simulator output.

    Camera Views
    ------------
    The LIBERO environment exposes two camera feeds in its observation:

    - **Third-person view** (``"full_image"``): a fixed overhead/front camera
      mounted in the scene. Provides global context about object positions,
      spatial relationships, and the robot's gross configuration.
    - **Wrist view** (``"wrist_image"``): a camera mounted on the robot's
      end-effector. Provides close-up detail about grasp alignment, contact
      geometry, and fine manipulation state — information that is largely
      invisible in the third-person frame.

    Both frames are extracted as NumPy ``uint8`` arrays of shape ``(H, W, 3)``
    (RGB channel order) and resized to the policy's expected input resolution
    via ``resize_image_for_policy``. The raw (un-resized) third-person frame
    is also returned separately for use in replay video generation.

    Proprioceptive State Vector
    ---------------------------
    The 8-dimensional state vector is formed by concatenating three components
    from the observation dictionary:

    +----------------------------------+--------+---------------------------------+
    | Source key                       | Shape  | Description                     |
    +==================================+========+=================================+
    | ``obs["robot0_eef_pos"]``        | ``(3,)`` | End-effector XYZ position in   |
    |                                  |          | world frame (metres)           |
    +----------------------------------+--------+---------------------------------+
    | ``quat2axisangle(``              | ``(3,)`` | End-effector orientation as    |
    | ``obs["robot0_eef_quat"]``)      |          | axis-angle vector (radians)    |
    +----------------------------------+--------+---------------------------------+
    | ``obs["robot0_gripper_qpos"]``   | ``(2,)`` | Gripper joint positions        |
    |                                  |          | (typically in [0, 1])          |
    +----------------------------------+--------+---------------------------------+

    The quaternion is converted to axis-angle (via ``quat2axisangle``) rather
    than used directly because the VLA was trained with axis-angle as the
    rotation representation. Using quaternion directly would cause an
    input-distribution mismatch and degrade policy performance.

    Parameters
    ----------
    obs : dict
        Raw observation dictionary returned by ``env.step()`` or
        ``env.set_init_state()``. Must contain at minimum the keys:
        ``"robot0_eef_pos"``, ``"robot0_eef_quat"``, ``"robot0_gripper_qpos"``,
        and the internal camera buffer keys consumed by
        ``get_libero_image`` / ``get_libero_wrist_image``.
    resize_size : tuple of (int, int)
        Target ``(height, width)`` in pixels for both resized camera frames.
        Derived at startup via ``get_image_resize_size(cfg)`` and must match
        the resolution the VLA backbone was trained on.

    Returns
    -------
    observation : dict
        Keys:
        - ``"full_image"``  : ``np.ndarray`` of shape ``(H', W', 3)``, dtype
          ``uint8`` — resized third-person RGB frame.
        - ``"wrist_image"`` : ``np.ndarray`` of shape ``(H', W', 3)``, dtype
          ``uint8`` — resized wrist-camera RGB frame.
        - ``"state"``       : ``np.ndarray`` of shape ``(8,)``, dtype
          ``float64`` — concatenated proprioceptive state vector.
    img : np.ndarray
        Original (un-resized) third-person frame of shape ``(H, W, 3)``.
        Passed to ``save_rollout_video`` at the end of each episode to
        construct a full-resolution replay video without re-rendering.
    """
    # ── Extract raw RGB frames from the simulator observation ─────────────────
    img       = get_libero_image(obs)       # Third-person camera: (H, W, 3) uint8
    wrist_img = get_libero_wrist_image(obs) # Wrist-mounted camera: (H, W, 3) uint8

    # ── Resize both frames to the policy's expected input resolution ──────────
    img_resized       = resize_image_for_policy(img, resize_size)       # (H', W', 3)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size) # (H', W', 3)

    # ── Build the structured observation dict consumed by get_vla_action ──────
    observation = {
        "full_image":  img_resized,        # Third-person frame at policy resolution
        "wrist_image": wrist_img_resized,  # Wrist frame at policy resolution
        "state": np.concatenate((
            obs["robot0_eef_pos"],                       # (3,) XYZ end-effector position
            quat2axisangle(obs["robot0_eef_quat"]),      # (3,) axis-angle orientation
            obs["robot0_gripper_qpos"],                  # (2,) gripper joint positions
        )),  # Resulting shape: (8,)
    }

    # Return processed observation dict AND the original frame for video replay
    return observation, img


# =============================================================================


def process_action(action: np.ndarray, model_family: str) -> np.ndarray:
    """
    Post-process a raw predicted action before sending it to the LIBERO
    simulator.

    Two sequential sign and scale transformations are applied to reconcile
    the VLA model's training output convention with the action space expected
    by ``env.step()``. Both steps are applied unconditionally to the gripper
    dimension only; the Cartesian and rotation dimensions are passed through
    unchanged.

    Transformation Pipeline
    -----------------------

    **Step 1 — Gripper normalisation** (``normalize_gripper_action``)

    The action head outputs the gripper dimension as a continuous value in
    ``[0, 1]``:

    - ``0.0`` → fully closed (grip)
    - ``1.0`` → fully open (release)

    LIBERO's ``env.step()`` expects a binary gripper command in ``{−1, +1}``:

    - ``−1`` → close gripper
    - ``+1`` → open gripper

    With ``binarize=True``, ``normalize_gripper_action`` applies a threshold
    at ``0.5`` and maps values below to ``−1`` and values above to ``+1``.
    This binarisation is important: sending continuous values to the LIBERO
    gripper controller can cause oscillation around the threshold and slow
    or prevent successful grasps.

    **Step 2 — OpenVLA sign inversion** (``invert_gripper_action``)

    During OpenVLA training the gripper dimension was **sign-flipped** to
    unify action conventions across heterogeneous training datasets:

    - Training convention: ``+1`` → close, ``−1`` → open
    - LIBERO convention:   ``−1`` → close, ``+1`` → open

    After Step 1, the value is already in the LIBERO binary ``{−1, +1}``
    range but still carries the **training sign**. ``invert_gripper_action``
    multiplies the gripper dimension by ``−1``, restoring the LIBERO
    convention. Without this step the gripper would always act in the
    opposite direction of the intended command.

    This inversion is architecture-specific and only applied when
    ``model_family == "openvla"``. Future model families with different
    training conventions may require different corrections or none at all.

    Parameters
    ----------
    action : np.ndarray of shape ``(action_dim,)``
        Raw predicted action vector from the model. The last dimension (or
        a dedicated index) is the gripper command; all other dimensions are
        end-effector Cartesian velocity / rotation commands.
    model_family : str
        Model architecture identifier. Currently the sign inversion in Step 2
        is only applied when this equals ``"openvla"``.

    Returns
    -------
    action : np.ndarray of shape ``(action_dim,)``
        Post-processed action vector ready to be passed to ``env.step()``.
        The array is modified in-place by the utility functions and the same
        object is returned.
    """
    # Step 1: Binarise gripper dimension from continuous [0,1] → binary {-1, +1}
    action = normalize_gripper_action(action, binarize=True)

    # Step 2: Invert gripper sign to undo the OpenVLA training convention flip
    if model_family == "openvla":
        action = invert_gripper_action(action)  # Multiply gripper dim by -1

    return action  # Fully post-processed; safe to pass to env.step()


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
    Execute a single evaluation episode in the LIBERO simulation environment.

    Implements the complete closed-loop control cycle with open-loop action
    chunking and a deterministic physics warm-up phase. This is the innermost
    loop of the evaluation: it is called once per trial and returns a binary
    success signal plus the full replay buffer.

    Episode Structure
    -----------------
    The episode has two distinct phases:

    **Phase 1 — Physics Warm-Up** (steps ``0`` to ``cfg.num_steps_wait - 1``)

    Dummy (zero-velocity) actions are injected into the simulator for the
    first ``cfg.num_steps_wait`` steps (default: 10). During this phase:
    - No observations are collected or passed to the model.
    - No replay images are saved.
    - The simulator integrates gravity and contact dynamics, allowing objects
      placed slightly above surfaces to settle into stable resting positions.

    Skipping this phase would cause the policy to observe a transient,
    physically inconsistent scene on its first query, leading to poor
    initial action predictions.

    **Phase 2 — Active Control** (steps ``cfg.num_steps_wait`` onwards)

    The policy is queried in open-loop chunks:
    1. ``prepare_observation`` converts the raw simulator output into the
       VLA's expected input format (resized images + proprio state vector).
    2. When the action queue is empty, ``get_vla_action`` is called to
       generate a new chunk of ``cfg.num_open_loop_steps`` actions.
    3. One action is dequeued per step and post-processed by
       ``process_action`` (gripper normalisation + sign inversion).
    4. The processed action is sent to the simulator via ``env.step()``.
    5. If ``done=True`` is returned, the episode terminates immediately
       as a success.

    Action Chunking
    ---------------
    Rather than re-querying the model at every single step, the policy
    generates ``cfg.num_open_loop_steps`` actions at once and executes them
    sequentially before re-querying. This is the *action chunking* scheme
    from ACT (Zhao et al., 2023) and is used by OpenVLA-OFT to amortise the
    latency of the VLA forward pass across multiple steps, effectively
    decoupling the control frequency from the inference frequency.

    The ``deque`` is initialised with ``maxlen=cfg.num_open_loop_steps``.
    When the deque is empty the model is re-queried; actions are ``extend``-ed
    in and consumed one at a time with ``popleft``. This pattern ensures that
    if the model returns fewer actions than expected (e.g. due to a diffusion
    schedule returning a shorter sequence), the deque never over-fills.

    Step Budget
    -----------
    The total number of simulator steps per episode is:

        ``cfg.num_steps_wait + TASK_MAX_STEPS[cfg.task_suite_name]``

    The warm-up steps are included in the total budget counter (``t``) but
    not counted toward the active step budget. This means the policy always
    receives exactly ``TASK_MAX_STEPS`` active steps regardless of the
    warm-up duration.

    For ``libero_goal``, ``TASK_MAX_STEPS = 300`` active steps × 20 Hz = 15
    seconds of manipulation time, which is sufficient for all single-step
    manipulation tasks in that suite.

    Failure Handling
    ----------------
    Any Python exception raised during the episode (e.g., MuJoCo instability,
    NaN in the action, CUDA out-of-memory) is caught by the outer
    ``try/except`` block. The episode is recorded as a failure (``success=False``),
    the error is logged to both the console and the log file via
    ``log_message``, and execution continues to the next episode. This prevents
    a single crashed episode from aborting the entire evaluation.

    Parameters
    ----------
    cfg : GenerateConfig
        Evaluation configuration providing:
        ``model_family``, ``num_steps_wait``, ``num_open_loop_steps``,
        ``task_suite_name``, ``use_film``.
    env : OffScreenRenderEnv
        Instantiated LIBERO simulation environment for the current task.
        Must have been constructed from the correct BDDL file.
    task_description : str
        Natural-language instruction string passed as language conditioning
        to the VLA. In ablation mode this is the keyword-only truncated
        command (e.g. ``"stove"``), not the original full sentence.
    model : torch.nn.Module
        Loaded VLA backbone in evaluation mode.
    resize_size : tuple of (int, int)
        Target ``(H, W)`` resolution for policy input images.
    processor : transformers.ProcessorMixin or None, optional
        VLA tokenizer and image pre-processor. Must be provided for OpenVLA.
    action_head : torch.nn.Module or None, optional
        Continuous action prediction head; ``None`` if fused into the backbone.
    proprio_projector : torch.nn.Module or None, optional
        Proprioceptive state projector; ``None`` if ``use_proprio=False``.
    noisy_action_projector : torch.nn.Module or None, optional
        DDIM noise-step projector; ``None`` if ``use_diffusion=False``.
    initial_state : np.ndarray or similar or None, optional
        Pre-recorded simulator state loaded from a ``.pruned_init`` file.
        Passed directly to ``env.set_init_state()`` to reproduce a specific
        object configuration. If ``None``, the environment's default post-reset
        state is used (stochastic, dependent on the environment's own RNG).
    log_file : io.TextIOWrapper or None, optional
        Open writable file handle for the per-run ``.txt`` log. Error messages
        are written here in addition to the console logger.

    Returns
    -------
    success : bool
        ``True`` if the environment returned ``done=True`` before the step
        budget was exhausted (indicating the goal condition was satisfied).
        ``False`` if the budget expired without completion, or if an exception
        was raised during the episode.
    replay_images : list of np.ndarray
        Ordered list of un-resized third-person camera frames captured at
        each active step. Each element has the original simulator render
        resolution (``cfg.env_img_res × cfg.env_img_res``). Passed to
        ``save_rollout_video`` after the episode to produce an MP4 replay.
    """
    # ── Phase 1 setup: reset environment and apply initial state ──────────────
    env.reset()  # Reset all object poses, robot joints, and simulator clock to t=0

    if initial_state is not None:
        # Overwrite the post-reset state with the pre-recorded initial state.
        # This is essential for reproducible evaluation: each trial i across
        # all ablation variants starts from the exact same object configuration.
        obs = env.set_init_state(initial_state)
    else:
        # No initial state provided: use whatever state the environment
        # produced after reset (behaviour depends on the env's own RNG).
        obs = env.get_observation()

    # ── Action chunk queue initialisation ─────────────────────────────────────
    # Warn if the configured chunk size does not match the training constant.
    # Mismatches reduce performance because the model was trained to predict
    # exactly NUM_ACTIONS_CHUNK steps of future motion.
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not "
            f"match the NUM_ACTIONS_CHUNK {NUM_ACTIONS_CHUNK} constant defined in "
            f"prismatic.vla.constants! For best performance (in terms of both speed "
            f"and success rate), we recommend executing the full action chunk."
        )

    # Fixed-capacity FIFO queue: holds at most num_open_loop_steps pending actions.
    # When empty, the model is re-queried. Actions are consumed one per step.
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # ── Episode state initialisation ──────────────────────────────────────────
    t = 0              # Global step counter: counts both warm-up AND active steps
    replay_images = [] # Accumulates un-resized third-person frames for video export
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]  # Active step budget (e.g. 300)

    success = False  # Will be set to True only if env signals done=True

    # ── Main episode loop (warm-up + active control) ───────────────────────────
    try:
        # Outer loop runs for warm-up + active steps combined.
        # Total iterations = cfg.num_steps_wait + max_steps
        while t < max_steps + cfg.num_steps_wait:

            # ── Phase 1: Physics warm-up ───────────────────────────────────────
            # Inject dummy zero-velocity actions for the first num_steps_wait
            # steps. This lets objects settle under gravity before the policy
            # begins observing the scene.
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(
                    get_libero_dummy_action(cfg.model_family)  # Zero-vel action
                )
                t += 1
                continue  # Skip the rest of the loop body; do NOT query the model

            # ── Phase 2: Active control ────────────────────────────────────────

            # Convert raw simulator observation to VLA input format.
            # Returns the structured dict AND the raw frame for the replay buffer.
            observation, img = prepare_observation(obs, resize_size)

            # Append raw (un-resized) frame to replay buffer for video export.
            # The raw frame is used (not the resized one) to produce a
            # full-resolution replay video.
            replay_images.append(img)

            # ── Action chunk re-query ──────────────────────────────────────────
            # Re-query the model only when the action queue is exhausted.
            # This is the open-loop chunking mechanism: one VLA forward pass
            # produces num_open_loop_steps actions, amortising inference latency.
            if len(action_queue) == 0:
                actions = get_vla_action(
                    cfg,                           # Config: unnorm_key, model_family, etc.
                    model,                         # VLA backbone
                    processor,                     # Tokenizer + image processor
                    observation,                   # Structured obs dict (images + state)
                    task_description,              # Language conditioning (keyword or full)
                    action_head=action_head,                         # Optional separate head
                    proprio_projector=proprio_projector,             # Optional proprio MLP
                    noisy_action_projector=noisy_action_projector,   # Optional DDIM MLP
                    use_film=cfg.use_film,                           # FiLM conditioning flag
                )
                # Log the first action's leading 4 dimensions for the first 50
                # active steps to provide a quick sanity-check in the log file.
                # Logging is suppressed after step 50 to keep logs readable.
                if t < 50:
                    logger.info(f"Step {t}: action[0] = {actions[0][:4]}...")

                # Fill the queue with all predicted actions; FIFO order preserved.
                # deque.extend() appends to the right; popleft() consumes from left.
                action_queue.extend(actions)

            # ── Dequeue one action for this step ──────────────────────────────
            # Pop the oldest (leftmost) action from the chunk.
            action = action_queue.popleft()

            # ── Action post-processing ────────────────────────────────────────
            # Apply gripper normalisation and OpenVLA sign inversion so that
            # the action matches the LIBERO env's expected format.
            action = process_action(action, cfg.model_family)

            # ── Simulator step ────────────────────────────────────────────────
            # Execute the processed action. Returns the next observation,
            # scalar reward, terminal flag, and info dict.
            # action.tolist() converts the NumPy array to a plain Python list
            # as required by the LIBERO env's action interface.
            obs, reward, done, info = env.step(action.tolist())

            # ── Early termination on success ───────────────────────────────────
            # The LIBERO environment sets done=True as soon as the task's
            # BDDL goal conditions are satisfied. Stop immediately to avoid
            # wasting steps after the task is already complete.
            if done:
                success = True
                break  # Exit the while loop; episode concludes as a success

            t += 1  # Advance the global step counter for the next iteration

    except Exception as e:
        # Catch-all: log the error and treat this episode as a failure.
        # This ensures that a single MuJoCo instability or CUDA error does
        # not abort the entire evaluation run.
        log_message(f"Episode error: {e}", log_file)
        # success remains False; replay_images contains frames up to the crash

    return success, replay_images

def run_ablation_task(
    cfg: GenerateConfig,
    task_key: str,
    task_info: dict,
    task_suite,
    task,
    model,
    resize_size: tuple,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
) -> tuple:
    """
    Execute the full evaluation loop for a single keyword-shortcut ablation variant.

    This function is the per-variant workhorse of the ablation study. For a
    given truncated instruction (e.g. ``"stove"`` or ``"bowl plate"``), it:

    1. Loads the pre-recorded initial states for the base task.
    2. Instantiates the LIBERO simulation environment from the ablation BDDL
       file, which embeds the truncated instruction in its ``:language`` field.
    3. Iterates over ``cfg.num_trials_per_task`` episodes, each starting from
       a reproducible pre-recorded initial state.
    4. For each episode, calls ``run_episode`` with the **ablation command**
       (not the original full instruction) as the language conditioning.
    5. Saves a replay video and logs per-episode results.
    6. Returns aggregate statistics for inclusion in the summary table.

    Ablation Command vs. Original Command
    --------------------------------------
    The distinction between ``ablation_command`` and ``original_description``
    is the scientific core of this function:

    - ``original_description`` is the full natural-language instruction from
      the LIBERO-Goal task (e.g. ``"Turn on the stove"``). It is used only for
      logging and for looking up custom initial states in the JSON file — it is
      **never** passed to the model during ablation rollouts.
    - ``ablation_command`` is the truncated keyword string extracted from the
      custom BDDL file (e.g. ``"stove"``). This is the **only** language
      signal the VLA receives during the episode. All success/failure outcomes
      are therefore attributable solely to whether the model can act correctly
      given this degraded instruction.

    Initial State Handling
    ----------------------
    Two modes are supported via ``cfg.initial_states_path``:

    **DEFAULT mode** (``cfg.initial_states_path == "DEFAULT"``)
        Uses the standard LIBERO-bundled initial states obtained from
        ``task_suite.get_task_init_states(task_id)``. Episode ``i`` always
        uses ``initial_states[i]``, giving a fixed, deterministic mapping
        between trial index and starting configuration across all ablation
        variants. This is essential for a fair comparison: every variant sees
        the exact same sequence of object configurations.

    **Custom JSON mode** (any other path)
        Loads initial states from a pre-computed JSON file produced by an
        expert-demonstration collection script. Each state is keyed by
        ``original_description.replace(" ", "_")`` (task key) and
        ``f"demo_{episode_idx}"`` (episode key). Episodes where the expert
        demonstration failed (``"success": false``) are **skipped** entirely
        and not counted toward ``task_episodes``. This filter ensures that
        the policy is only evaluated on configurations where the task is
        known to be solvable.

    Video Replay Naming
    -------------------
    Each episode's replay video filename encodes the task ID, the ablation
    variant key, and the command string (spaces replaced by underscores) so
    that videos from different variants are unambiguously distinguishable in
    the output directory:

        ``ablation_task{N}_{task_key}_{ablation_command_with_underscores}``

    Parameters
    ----------
    cfg : GenerateConfig
        Evaluation configuration providing: ``ablation_task_id``,
        ``num_trials_per_task``, ``initial_states_path``, ``env_img_res``,
        ``model_family``, ``use_film``, ``use_wandb``.
    task_key : str
        Short identifier for this ablation variant as defined in
        ``ABLATION_CONFIGS`` (e.g. ``"stove1"``, ``"bowl_plate3"``).
        Used in log headers, W&B metric names, and video filenames.
    task_info : dict
        Ablation variant configuration dict with two keys:
        - ``"bddl_file"`` : str — filename of the custom BDDL file whose
          ``:language`` field contains the truncated instruction.
        - ``"expected_command"`` : str — the keyword-only instruction string
          (used for reference/logging; the actual command is extracted live
          from the BDDL via ``get_libero_env``).
    task_suite : libero.libero.benchmark.TaskSuite
        Instantiated LIBERO task suite (e.g. ``libero_goal``). Used by
        ``load_initial_states`` to retrieve pre-recorded initial states.
    task : libero.libero.benchmark.Task
        The base LIBERO task object for ``cfg.ablation_task_id``. Its
        ``language`` attribute is the original full instruction. Passed to
        ``get_libero_env`` alongside the ablation BDDL override.
    model : torch.nn.Module
        Loaded VLA backbone in evaluation mode.
    resize_size : tuple of (int, int)
        Target ``(H, W)`` resolution for policy input images.
    processor : transformers.ProcessorMixin or None, optional
        VLA tokenizer and image pre-processor.
    action_head : torch.nn.Module or None, optional
        Continuous action prediction head.
    proprio_projector : torch.nn.Module or None, optional
        Proprioceptive state projector.
    noisy_action_projector : torch.nn.Module or None, optional
        DDIM noise-step conditioning projector.
    log_file : io.TextIOWrapper or None, optional
        Open writable file handle for the per-run ``.txt`` log.

    Returns
    -------
    task_success_rate : float
        Fraction of completed episodes in which ``done=True`` was reached
        before the step budget expired. Value in ``[0.0, 1.0]``.
        Returns ``0.0`` if zero episodes were completed (e.g. all skipped).
    task_episodes : int
        Number of episodes actually executed and counted. May be less than
        ``cfg.num_trials_per_task`` if some episodes were skipped due to
        failed expert demonstrations in custom JSON mode.
    task_successes : int
        Absolute number of successful episodes out of ``task_episodes``.
    """
    task_id = cfg.ablation_task_id  # 0-indexed task ID; used for state loading and logging

    # ── Section header in the log file ────────────────────────────────────────
    log_message("=" * 80, log_file)
    log_message(f"ABLATION TASK: {task_key.upper()}", log_file)   # e.g. "STOVE1"
    log_message(f"BDDL File: {task_info['bddl_file']}", log_file) # Custom BDDL filename
    log_message("=" * 80, log_file)

    # ── Load pre-recorded initial simulator states ─────────────────────────────
    # Returns (initial_states, all_initial_states).
    # In DEFAULT mode: all_initial_states is None.
    # In custom JSON mode: all_initial_states is the full parsed JSON dict.
    initial_states, all_initial_states = load_initial_states(
        cfg, task_suite, task_id, log_file
    )

    # ── Instantiate environment from the ablation BDDL file ───────────────────
    # get_libero_env is called with ablation_bddl_file to override the default
    # BDDL for this task. It returns:
    #   env                 : OffScreenRenderEnv built from the custom BDDL
    #   task_description    : language string extracted from the ablation BDDL
    #                         (the truncated keyword command, e.g. "stove")
    #   original_description: language string from the base task BDDL
    #                         (the full instruction, e.g. "Turn on the stove")
    env, task_description, original_description = get_libero_env(
        task,
        ablation_bddl_file=task_info['bddl_file'],  # Override BDDL with ablation file
        resolution=cfg.env_img_res                   # MuJoCo render resolution (pixels)
    )

    # The ablation command is the truncated instruction extracted from the BDDL.
    # This is what the VLA receives as language conditioning during all rollouts.
    ablation_command = task_description

    # Log both commands for side-by-side comparison in the analysis output
    log_message(
        f"Original Task {task_id+1} Command: {original_description}", log_file
    )
    log_message(
        f"Ablation Command (from BDDL): '{ablation_command}'", log_file
    )

    # ── Episode loop ──────────────────────────────────────────────────────────
    task_episodes  = 0  # Episodes actually executed (may be < num_trials_per_task
                        # if some are skipped in custom JSON mode)
    task_successes = 0  # Episodes where done=True was received before budget expired

    for episode_idx in tqdm.tqdm(
        range(cfg.num_trials_per_task),
        desc=f"Ablation {task_key}",  # Progress bar label in stdout
    ):

        # ── Initial state selection ────────────────────────────────────────────
        if cfg.initial_states_path == "DEFAULT":
            # Standard mode: use the i-th pre-recorded LIBERO initial state.
            # Index is consistent across all ablation variants, guaranteeing
            # that each variant is tested on the same sequence of scenes.
            initial_state = initial_states[episode_idx]
        else:
            # Custom JSON mode: look up the initial state by task name and
            # episode index in the pre-computed expert-demonstration JSON file.
            initial_states_task_key = original_description.replace(" ", "_")
            # e.g. "Turn_on_the_stove"
            episode_key = f"demo_{episode_idx}"  # e.g. "demo_0", "demo_1", ...

            # Skip episodes where the expert demonstration itself failed.
            # A policy should not be penalised for scenarios where even the
            # oracle could not complete the task.
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(
                    f"Skipping task {task_id} episode {episode_idx} due to "
                    f"failed expert demo!",
                    log_file,
                )
                continue  # Do not increment task_episodes; move to next index

            # Extract the initial state array from the nested JSON structure
            initial_state = np.array(
                all_initial_states[initial_states_task_key][episode_key]["initial_state"]
            )  # Converts list-of-floats from JSON back to np.ndarray

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # ── Execute episode with the ablation (truncated) command ──────────────
        # Critically: ablation_command (not original_description) is passed as
        # task_description. The model receives only the keyword-only instruction.
        success, replay_images = run_episode(
            cfg,
            env,
            ablation_command,        # ← Truncated keyword command (the ablation)
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,           # Deterministic starting configuration
            log_file,
        )

        # ── Update episode and success counters ────────────────────────────────
        task_episodes += 1           # Count this episode as completed
        if success:
            task_successes += 1      # Count as a success only if done=True received

        # ── Save replay video ─────────────────────────────────────────────────
        # Filename encodes task ID, variant key, and ablation command so videos
        # from different ablation conditions are unambiguously distinguishable.
        save_rollout_video(
            {'image': replay_images},   # Dict wrapper expected by save_rollout_video
            task_episodes,              # Episode number (1-indexed) for filename suffix
            success=success,            # Appended to filename for quick visual triage
            task_description=(
                f"ablation_task{task_id+1}_{task_key}_"
                f"{ablation_command.replace(' ', '_')}"
            ),                          # Encodes variant identity in filename
            log_file=log_file,
            change_command=True,        # Flag signalling this is a command-variant run
            command_level="ablation",   # Sub-directory level for video organisation
        )

        # ── Per-episode logging ────────────────────────────────────────────────
        log_message(f"Success: {success}", log_file)
        log_message(
            f"# episodes completed so far: {task_episodes}", log_file
        )
        log_message(
            f"# successes: {task_successes} "
            f"({task_successes / task_episodes * 100:.1f}%)",
            log_file,
        )

    # ── Aggregate task statistics ──────────────────────────────────────────────
    # Guard against zero-episode edge case (e.g. all episodes skipped in JSON mode)
    task_success_rate = (
        float(task_successes) / float(task_episodes)
        if task_episodes > 0
        else 0.0
    )

    log_message(f"Current task success rate: {task_success_rate}", log_file)

    # ── Optional W&B metric streaming ─────────────────────────────────────────
    # Each variant gets its own W&B key so the dashboard can plot all variants
    # as separate lines on the same chart for easy comparison.
    if cfg.use_wandb:
        wandb.log({
            f"success_rate/ablation_{task_key}": task_success_rate,  # e.g. success_rate/ablation_stove1
            f"num_episodes/ablation_{task_key}": task_episodes,       # e.g. num_episodes/ablation_stove1
        })

    return task_success_rate, task_episodes, task_successes


# =============================================================================


def print_ablation_results(
    results: dict,
    task_name: str,
    ablation_tests: dict,
) -> None:
    """
    Print a formatted ASCII summary table of keyword-shortcut ablation results
    to stdout.

    The table presents one row per ablation variant, showing the keyword
    command tested, the number of successful and total episodes, and the
    resulting success rate as a percentage. A final row shows the arithmetic
    mean success rate across all variants.

    Table Format
    ------------
    ::

        ====================================================================================================
        ABLATION STUDY RESULTS - Task: Turn on the stove
        ====================================================================================================
        Test                 | Command              | Success Rate |        Episodes
        ----------------------------------------------------------------------------------------------------
        stove1               | stove                |       32.0%  |    16/50
        stove2               | bowl stove           |       48.0%  |    24/50
        stove3               | plate stove          |       46.0%  |    23/50
        stove4               | Turn on              |       10.0%  |     5/50
        ====================================================================================================
        AVERAGE              |                      |       34.0%  |    68/200
        ====================================================================================================

    Column widths are fixed:

    - ``Test``         : 20 characters (left-aligned)
    - ``Command``      : 20 characters (left-aligned)
    - ``Success Rate`` : 12 characters (right-aligned, ``XX.X%`` format)
    - ``Episodes``     : 15 characters total; displayed as ``successes/total``

    Interpretation Guide
    --------------------
    When reading the results table, the following patterns indicate specific
    failure modes:

    - **High success rate for single-noun variants** (e.g. ``"stove"`` alone
      ≈ full instruction): strong evidence of a keyword shortcut. The model
      does not need the full instruction to execute the correct motion.
    - **Low success rate for verb-only variants** (e.g. ``"Turn on"`` much
      lower than ``"stove"``): the model ignores the action verb and relies
      primarily on the object noun for task disambiguation.
    - **Similar rates for canonical and reversed word-order** (e.g.
      ``"bowl plate"`` ≈ ``"plate bowl"``): the model is not tracking
      the syntactic role (subject vs. target) of each object noun.
    - **Uniformly low rates across all variants**: the model does use the
      full instruction, but lacks compositional understanding of keywords alone.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping from ablation variant key (e.g. ``"stove1"``) to a result
        dict with keys:
        - ``"success_rate"`` : float — fraction of successful episodes in [0,1].
        - ``"successes"``    : int   — absolute number of successful episodes.
        - ``"episodes"``     : int   — total episodes executed for this variant.
    task_name : str
        Human-readable name of the base task (e.g. ``"Turn on the stove"``).
        Printed in the table header for identification.
    ablation_tests : dict[str, dict]
        Full ablation test registry for this task (from ``get_ablation_tasks``).
        Used to retrieve ``"expected_command"`` for each variant key so the
        table can display the actual keyword string tested.

    Returns
    -------
    None
        Output is written to stdout only; nothing is returned.
    """
    # ── Table header ──────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"ABLATION STUDY RESULTS - Task: {task_name}")
    print("=" * 100)

    # Column header row with fixed-width alignment
    print(
        f"{'Test':<20} | {'Command':<20} | {'Success Rate':>12} | {'Episodes':>15}"
    )
    print("-" * 100)

    # ── Per-variant result rows ────────────────────────────────────────────────
    for task_key, result in results.items():
        cmd   = ablation_tests[task_key]['expected_command']  # Keyword string
        sr    = result['success_rate']   # Float in [0, 1]
        succ  = result['successes']      # Absolute success count
        total = result['episodes']       # Total episode count

        # Format: left-aligned key + command, right-aligned rate and episode counts
        print(
            f"{task_key:<20} | {cmd:<20} | {sr:>11.1%} | "
            f"{succ:>6}/{total:<7}"
            # e.g. "stove1               | stove                |       32.0% |     16/50    "
        )

    print("=" * 100)

    # ── Summary row: arithmetic mean across all variants ──────────────────────
    # Arithmetic mean (not weighted by episode count) is used so that variants
    # with skipped episodes do not dominate the aggregate.
    avg_sr     = sum(r['success_rate'] for r in results.values()) / len(results)
    total_succ = sum(r['successes'] for r in results.values())   # Grand total successes
    total_eps  = sum(r['episodes']  for r in results.values())   # Grand total episodes

    print(
        f"{'AVERAGE':<20} | {'':<20} | {avg_sr:>11.1%} | "
        f"{total_succ:>6}/{total_eps:<7}"
    )
    print("=" * 100)


# =============================================================================


@draccus.wrap()
def eval_ablation(cfg: GenerateConfig) -> float:
    """
    Main entry point for the keyword-shortcut ablation evaluation study.

    Decorated with ``@draccus.wrap()`` so that all ``GenerateConfig`` fields
    are automatically parsed from CLI flags and/or a YAML config file. When
    invoked as ``python run_libero_ablation.py --pretrained_checkpoint ...``,
    draccus populates ``cfg`` and calls this function.

    Execution Flow
    --------------
    The function orchestrates the full ablation pipeline in eight sequential
    phases:

    1. **Debug setup** — if ``cfg.debug=True``, a debugpy listener is opened
       on port 5678 and execution blocks until a remote debugger attaches.
       Useful for interactive debugging on a SLURM compute node.

    2. **Configuration validation** — ``validate_config`` checks for invalid
       flag combinations (missing checkpoint, quantisation conflicts, wrong
       task suite, unknown ``ablation_task_id``) before any heavyweight
       resource is loaded.

    3. **Random seed** — ``set_seed_everywhere`` seeds Python, NumPy, and
       PyTorch globally to make all stochastic operations (model sampling,
       environment RNG) deterministic and reproducible across runs.

    4. **Ablation configuration** — ``get_ablation_tasks`` retrieves the
       task-specific registry of keyword test variants (BDDL filenames and
       expected command strings).

    5. **Model initialisation** — ``initialize_model`` loads the VLA backbone,
       optional action head, proprioceptive projector, and DDIM projector.

    6. **LIBERO task setup** — the LIBERO benchmark registry is queried via
       ``benchmark.get_benchmark_dict()`` to obtain the base ``Task`` object
       for ``cfg.ablation_task_id``. This is needed to (a) construct the
       environment via ``get_libero_env`` and (b) retrieve initial states.

    7. **Logging setup** — ``setup_logging`` opens the per-run ``.txt`` log
       file and optionally initialises a W&B run.

    8. **Ablation loop** — for each variant in ``ablation_tests``,
       ``run_ablation_task`` is called, which executes the full episode
       rollout loop and returns per-variant statistics. Results are
       collected in a ``dict`` keyed by variant name.

    After the loop:
    - ``print_ablation_results`` renders the ASCII summary table to stdout.
    - Per-variant and aggregate results are written to the log file.
    - If W&B is enabled, the arithmetic mean success rate and the log file
      are uploaded.
    - The log file is closed.
    - The arithmetic mean success rate is returned (for programmatic use).

    Return Value
    ------------
    The return value is the arithmetic mean success rate across all ablation
    variants (a float in ``[0.0, 1.0]``). This is used by draccus to forward
    the result to any calling context (e.g., a hyperparameter sweep driver).

    Parameters
    ----------
    cfg : GenerateConfig
        Fully populated configuration object. All fields are set by draccus
        from CLI flags and/or a YAML file before this function is called.

    Returns
    -------
    avg_sr : float
        Arithmetic mean of per-variant success rates across all ablation
        tests. Value in ``[0.0, 1.0]``.
    """
    # ── Phase 1: Optional remote debugger setup ────────────────────────────────
    if cfg.debug:
        import debugpy  # Lazy import: only required when debug mode is active
        debugpy.listen(('0.0.0.0', 5678))  # Listen on all interfaces, port 5678
        print("Waiting for debugger attach")  # Blocks stdout; user attaches VS Code
        debugpy.wait_for_client()            # Execution pauses here until a client connects

    # ── Phase 2: Configuration validation (fail fast before heavy I/O) ────────
    validate_config(cfg)  # Raises AssertionError or ValueError on any invalid field

    # ── Phase 3: Global random seed ────────────────────────────────────────────
    # Must be called before any model weight loading, environment reset, or
    # NumPy/PyTorch operation to ensure fully deterministic behaviour.
    set_seed_everywhere(cfg.seed)

    # ── Phase 4: Ablation task configuration ───────────────────────────────────
    ablation_config = get_ablation_tasks(cfg.ablation_task_id)
    task_name     = ablation_config['task_name']   # e.g. "Turn on the stove"
    ablation_tests = ablation_config['tests']       # Dict of variant_key → {bddl_file, expected_command}

    # ── Phase 5: Model and component initialisation ────────────────────────────
    model, action_head, proprio_projector, noisy_action_projector, processor = (
        initialize_model(cfg)   # Loads VLA backbone + optional heads onto GPU(s)
    )

    # Derive the (H, W) resolution the policy expects for its image inputs.
    # This depends on the model family and is used in prepare_observation.
    resize_size = get_image_resize_size(cfg)

    # ── Phase 6: LIBERO task suite and base task setup ─────────────────────────
    # Retrieve the full benchmark registry: maps suite names → task suite factories
    benchmark_dict = benchmark.get_benchmark_dict()

    # Instantiate the libero_goal task suite object (contains 10 tasks + init states)
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # Get the base Task object for the target task ID.
    # This gives us: task.language (full instruction), task.bddl_file, etc.
    # Note: get_libero_env will override the BDDL file with the ablation version.
    task = task_suite.get_task(cfg.ablation_task_id)

    # ── Phase 7: Logging setup ─────────────────────────────────────────────────
    # Opens the per-run .txt log file and optionally initialises a W&B run.
    log_file, local_log_filepath, run_id = setup_logging(cfg, task_name)

    # Log the full run configuration header for reproducibility documentation
    log_message("=" * 80, log_file)
    log_message(
        f"ABLATION STUDY: Task {cfg.ablation_task_id+1} Keyword Shortcut Analysis",
        log_file,
    )
    log_message("=" * 80, log_file)
    log_message(f"Task Name: {task_name}", log_file)
    log_message(f"Base Command: {task.language}", log_file)      # Full original instruction
    log_message(f"Model: {cfg.pretrained_checkpoint}", log_file)  # Checkpoint path
    log_message(f"Seed: {cfg.seed}", log_file)
    log_message(f"Trials per ablation: {cfg.num_trials_per_task}", log_file)
    log_message(f"Ablation tests: {list(ablation_tests.keys())}", log_file)  # e.g. ['stove1', ...]
    log_message("=" * 80, log_file)

    # ── Phase 8: Ablation variant loop ────────────────────────────────────────
    results = {}  # Accumulates {variant_key: {success_rate, episodes, successes}}

    for task_key, task_info in ablation_tests.items():
        # Run the full episode loop for this variant (e.g. "stove1" with cmd "stove")
        sr, episodes, successes = run_ablation_task(
            cfg,
            task_key,           # Variant identifier (e.g. "stove1")
            task_info,          # Variant config: {bddl_file, expected_command}
            task_suite,         # Needed by load_initial_states
            task,               # Base task object; passed to get_libero_env
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
        )

        # Store per-variant statistics for the summary table and final log
        results[task_key] = {
            'success_rate': sr,         # Float in [0, 1]
            'episodes':     episodes,   # Int: episodes actually executed
            'successes':    successes,  # Int: successful episodes
        }

    # ── Results: ASCII summary table (stdout) ─────────────────────────────────
    print_ablation_results(results, task_name, ablation_tests)

    # ── Results: Per-variant lines in the log file ─────────────────────────────
    log_message("\n" + "=" * 80, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 80, log_file)

    for task_key, result in results.items():
        cmd = ablation_tests[task_key]['expected_command']  # Keyword tested
        log_message(
            f"{task_key} ('{cmd}'): {result['success_rate']:.1%} "
            f"({result['successes']}/{result['episodes']})",
            log_file,
        )

    # Compute arithmetic mean success rate across all variants
    avg_sr = sum(r['success_rate'] for r in results.values()) / len(results)
    log_message(f"\nAVERAGE: {avg_sr:.1%}", log_file)
    log_message("=" * 80, log_file)

    # ── Optional W&B final upload ──────────────────────────────────────────────
    if cfg.use_wandb:
        wandb.log({"average_success_rate": avg_sr})  # Aggregate scalar metric
        wandb.save(local_log_filepath)               # Upload the .txt log as an artifact

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if log_file:
        log_file.close()  # Flush and release the file handle

    return avg_sr  # Returned to draccus / calling context


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # draccus.wrap() on eval_ablation handles all CLI argument parsing.
    # Calling eval_ablation() here triggers draccus to read sys.argv,
    # populate GenerateConfig, and invoke the decorated function.
    eval_ablation()