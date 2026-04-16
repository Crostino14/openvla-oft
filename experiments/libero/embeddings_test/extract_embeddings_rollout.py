"""
extract_embeddings_rollout.py
==============================

Extract multimodal hidden-state embeddings from a fine-tuned OpenVLA model
during **real inference rollouts** on LIBERO manipulation tasks.

Scientific Motivation
---------------------
Standard embedding analysis approaches extract representations from a frozen
model using a single synthetic forward pass with a fixed or random image.
This script takes a fundamentally different approach: embeddings are extracted
from the model's internal state **during actual task execution**, where the
robot is interacting with a live MuJoCo simulation, observing real scene
states, and executing real predicted actions.

This grounding matters because:

1. **Visual context is task-relevant**: the model observes the actual scene
   at each step — objects in their true positions, the robot arm in its current
   configuration — rather than a static placeholder image. The resulting
   embedding reflects the intersection of the instruction and the current
   visual state, not just the language in isolation.

2. **Temporal dynamics are captured**: in full rollout mode, embeddings are
   collected at every step and averaged, yielding a trajectory-level
   representation that summarises how the model "sees" the task across the
   entire episode. This is more informative than a single snapshot.

3. **Behaviour-aligned representations**: because the same forward pass
   produces both the embedding and the policy's action distribution, the
   representation is directly tied to the model's decision-making process.

Two extraction modes trade off fidelity against compute:

**full_rollout mode** (default)
    Runs the policy for up to ``TASK_MAX_STEPS`` steps. At each active step,
    a full multimodal forward pass is executed and the resulting hidden state
    is recorded. The per-rollout embedding is the temporal mean over all step
    embeddings. Additionally, per-rollout success flags, step counts, and all
    raw step embeddings are saved for downstream analysis.

**first_step_only mode**
    The environment is reset and the physics warm-up runs, but no actions are
    predicted or executed after warm-up. A single embedding is extracted from
    the first observation the policy would act upon. This isolates the model's
    *prior* representation — how it encodes the task before any visual feedback
    from its own motion — and runs ~20× faster than full rollout mode.

Embedding Extraction Mechanism
-------------------------------
The standard OpenVLA ``predict_action()`` API does not expose per-step hidden
states. This script therefore re-implements the multimodal forward pass
manually inside ``extract_embedding_from_forward()``:

1. The processor tokenises the prompt and pre-processes the image(s).
2. The vision backbone extracts patch embeddings from the camera frame(s).
3. The projector maps patch embeddings into the LLM's embedding space.
4. A multimodal sequence is assembled:
   ``[BOS] | [visual patches (N)] | [text tokens (L-1)]``
5. An extended causal attention mask is built covering all positions.
6. The language model runs one forward pass with ``output_hidden_states=True``.
7. The last-layer hidden states at **text-only positions** (BOS + text tokens;
   visual patches are excluded) are mean-pooled into a single vector of
   shape ``(hidden_dim,)``.

Why text positions only? Visual patch hidden states carry low-level spatial
information that fluctuates step-to-step. Text-position states accumulate a
holistic, task-relevant summary of the current visual scene conditioned on the
instruction — a more stable and interpretable representation.

Output Format
-------------
All results are serialised as a single pickle file containing
``dict[str, dict]`` keyed by ``"task_{N:02d}_{level}"``. See the module-level
docstring in the complete file for the full field schema.

Dependencies
------------
- torch          : GPU inference, tensor operations
- numpy          : Array accumulation and statistics
- PIL            : Image format conversion for the processor
- pickle         : Binary serialisation of the embedding dictionary
- libero.libero  : LIBERO benchmark registry and environment factory
- prismatic      : OpenVLA model and constants

Author: Agostino Cardamone
"""


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import os       # Path construction (os.path.join), directory creation (os.makedirs),
                # existence checks (os.path.exists)
import sys      # sys.path manipulation to make local packages importable

import pickle   # Binary serialisation and deserialisation of Python objects
                # (used to save the final embedding dictionary as a .pkl file)

import numpy as np  # Numerical array operations: stacking, mean, concatenation, dtype casting

from pathlib import Path   # Cross-platform object-oriented file path construction;
                            # used to resolve the absolute path of this script

from PIL import Image      # Pillow library: converts raw NumPy uint8 RGB arrays to
                            # PIL.Image.Image objects, which the OpenVLA processor expects

from collections import deque  # Fixed-capacity double-ended queue used as a FIFO action
                                # buffer for open-loop action chunking in run_single_episode

from dataclasses import dataclass   # Decorator that auto-generates __init__, __repr__,
                                     # and __eq__ from annotated class fields
from dataclasses import field        # Helper for dataclass default_factory values (imported
                                     # for completeness; not directly used in this excerpt)

from typing import Optional    # Type hint for values that may be None
from typing import List        # Type hint for typed lists (Python < 3.9 compatibility)
from typing import Dict        # Type hint for typed dicts (Python < 3.9 compatibility)
from typing import Any         # Type hint for untyped / heterogeneous values


# =============================================================================
# PROJECT ROOT PATH RESOLUTION
# =============================================================================

# Resolve the absolute path of this script, independent of the working directory
current_file = Path(__file__).resolve()

# Navigate three directory levels up to reach the repository root (openvla-oft/)
# Assumes the script lives at: openvla-oft/experiments/libero/analysis/
project_root = current_file.parent.parent.parent

# Prepend the project root to sys.path so that sibling top-level packages
# (experiments/, prismatic/) shadow any installed versions with the same names
sys.path.insert(0, str(project_root))


# =============================================================================
# INTERNAL UTILITIES — OPENVLA MODEL COMPONENTS
# =============================================================================

from experiments.openvla_utils import (
    get_processor,              # Returns the combined tokenizer + image pre-processor
                                # for the OpenVLA model family
    get_vla,                    # Instantiates the full VLA model (backbone + projector + LLM)
                                # from a local checkpoint path or HuggingFace Hub ID
    get_action_head,            # Loads the continuous action prediction head (L1 or DDIM)
                                # from a separate weight file alongside the checkpoint
    get_proprio_projector,      # Loads the MLP that projects the 8-dim proprioceptive state
                                # vector into the LLM's hidden embedding space
    get_noisy_action_projector, # Loads the noise-step conditioning projector used by the
                                # DDIM diffusion action head at inference time
    get_vla_action,             # Executes the full VLA inference pipeline and returns
                                # a predicted action chunk of shape (num_open_loop_steps, 7)
    resize_image_for_policy,    # Resizes a raw NumPy uint8 image to the (H, W) resolution
                                # expected by the policy's image pre-processor
)


# =============================================================================
# INTERNAL UTILITIES — LIBERO ENVIRONMENT AND TASK HELPERS
# =============================================================================

from experiments.libero.libero_utils import (
    extract_command_from_bddl,  # Parses the `:language "..."` field from a BDDL file
                                 # and returns the instruction string
    get_libero_path,            # Returns the absolute path to a named LIBERO resource
                                 # directory (e.g. "bddl_files", "init_states")
    get_libero_env,             # Factory function: creates a LIBERO OffScreenRenderEnv
                                 # for a given task, with optional command substitution
    get_libero_image,           # Extracts the third-person RGB camera frame from a
                                 # raw LIBERO observation dictionary
    get_libero_wrist_image,     # Extracts the wrist-mounted RGB camera frame from a
                                 # raw LIBERO observation dictionary
    get_libero_dummy_action,    # Returns a zero-action vector of the correct dimension
                                 # for the given model family (used during warm-up phase)
    quat2axisangle,             # Converts a quaternion (qx, qy, qz, qw) to a 3D
                                 # axis-angle representation (ax, ay, az)
)


# =============================================================================
# INTERNAL UTILITIES — GENERAL ROBOT EXPERIMENT HELPERS
# =============================================================================

from experiments.robot_utils import (
    get_image_resize_size,    # Derives the target (H, W) image resolution from the config;
                              # typically (224, 224) or (256, 256) depending on the backbone
    invert_gripper_action,    # Flips the sign of the gripper dimension to undo the sign
                              # convention adopted during OpenVLA training data collection
    normalize_gripper_action, # Remaps the gripper dimension from the continuous [0, 1]
                              # model output range to the {-1, +1} binary range expected
                              # by the LIBERO simulator's env.step() interface
)


# =============================================================================
# LIBERO BENCHMARK REGISTRY
# =============================================================================

from libero.libero import benchmark  # Provides benchmark.get_benchmark_dict(), which
                                      # returns a dict mapping suite name strings
                                      # (e.g. "libero_goal") to benchmark class factories


# =============================================================================
# OPENVLA ARCHITECTURE CONSTANT
# =============================================================================

from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # Integer constant defining how many
                                                         # actions the model predicts per
                                                         # forward pass (e.g. 8). Used to
                                                         # set the action queue capacity and
                                                         # to validate cfg.num_open_loop_steps


# =============================================================================
# PER-SUITE STEP BUDGET
# =============================================================================

TASK_MAX_STEPS = {
    "libero_spatial": 220,  # Spatial reasoning tasks — short horizon, mostly near-reach
    "libero_object":  280,  # Object manipulation tasks — slightly longer than spatial
    "libero_goal":    200,  # Goal-conditioned 10-task suite — moderate horizon
    "libero_10":      520,  # Long-horizon 10-task suite — up to 10 sub-goals per task
    "libero_90":      400,  # Large-scale 90-task suite — heterogeneous horizon lengths
}
"""
dict[str, int]
    Maps each LIBERO task suite name to the per-episode maximum number of
    **active** simulator steps (excluding the physics warm-up phase).
    Episodes that reach this limit without a ``done=True`` signal are
    terminated and recorded as failures.

    These values are calibrated to be generous enough for a competent policy
    to succeed on most tasks while preventing excessively long runs that would
    inflate wall-clock time on a compute cluster.
"""


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class EmbeddingConfig:
    """
    Configuration schema for embedding extraction during rollout inference.

    This dataclass serves a dual purpose:

    1. **Model loading interface**: the fields mirror those of the standard
       ``GenerateConfig`` used in ``run_libero_eval.py``, ensuring that shared
       model-loading helpers (``get_vla``, ``get_processor``, ``get_action_head``,
       etc.) receive the same interface they expect without modification.

    2. **Extraction settings**: additional fields control the embedding
       extraction mode, environment resolution, and task suite selection.

    All fields have sensible defaults and can be overridden via the CLI
    argument parser in the ``__main__`` block.

    Attribute Groups
    ----------------

    Model Parameters
    ~~~~~~~~~~~~~~~~
    pretrained_checkpoint : str, default ``""``
        Local directory path or HuggingFace Hub model ID for the fine-tuned
        OpenVLA checkpoint. **Must be set before use** — an empty string will
        cause ``get_vla()`` to raise an error.
    model_family : str, default ``"openvla"``
        Model family identifier. Controls:
        - Which processor is loaded (``get_processor``).
        - Whether gripper sign inversion is applied in ``process_action``.
        Currently only ``"openvla"`` is fully supported.

    Action Head Parameters
    ~~~~~~~~~~~~~~~~~~~~~~
    use_l1_regression : bool, default ``True``
        Load and use the continuous action head trained with an L1 regression
        objective. Controls whether ``get_action_head`` is called.
        Mutually exclusive with ``use_diffusion``.
    use_diffusion : bool, default ``False``
        Load and use the DDIM diffusion action head instead of L1 regression.
        When ``True``, also loads the ``noisy_action_projector``.
        Mutually exclusive with ``use_l1_regression``.
    num_diffusion_steps : int, default ``50``
        Number of DDIM denoising iterations per action prediction. Higher
        values improve action quality at the cost of increased inference latency.
        Only relevant when ``use_diffusion=True``.
    use_film : bool, default ``False``
        Apply FiLM (Feature-wise Linear Modulation) conditioning, which injects
        language features into visual backbone intermediate layers. Must match
        the architecture used during fine-tuning.
    num_images_in_input : int, default ``2``
        Number of camera views fed to the VLA. ``2`` = third-person + wrist
        camera (standard for LIBERO). ``1`` = third-person only. Controls
        whether the wrist image is processed and concatenated in
        ``extract_embedding_from_forward``.
    use_proprio : bool, default ``True``
        Include the 8-dimensional proprioceptive state vector as additional
        conditioning. When ``True``, ``get_proprio_projector`` is called to
        load the projector MLP.

    Inference Parameters
    ~~~~~~~~~~~~~~~~~~~~
    center_crop : bool, default ``True``
        Apply deterministic centre cropping to policy input images at eval
        time. Should match the augmentation used during fine-tuning.
    num_open_loop_steps : int, default ``8``
        Number of actions from each predicted chunk to execute before re-querying
        the model. Should equal ``NUM_ACTIONS_CHUNK`` from the training config;
        a mismatch will produce a warning in ``run_single_episode``.

    Quantisation
    ~~~~~~~~~~~~
    load_in_8bit : bool, default ``False``
        Load model weights in INT8 format via ``bitsandbytes``. Reduces GPU
        memory by approximately 50% at a small accuracy cost.
        **Mutually exclusive with** ``load_in_4bit``.
    load_in_4bit : bool, default ``False``
        Load model weights in NF4 format via ``bitsandbytes``. Reduces GPU
        memory by approximately 75% at a larger accuracy cost.
        **Mutually exclusive with** ``load_in_8bit``.

    Task Parameters
    ~~~~~~~~~~~~~~~
    task_suite_name : str, default ``"libero_goal"``
        LIBERO benchmark suite identifier. Used to:
        - Look up the benchmark task registry (``benchmark.get_benchmark_dict()``).
        - Resolve the action un-normalisation key in ``model.norm_stats``.
        - Look up the per-suite step budget in ``TASK_MAX_STEPS``.
    unnorm_key : str, default ``""``
        Action un-normalisation key for de-standardising model outputs.
        Resolved automatically by ``load_model_and_components()`` from
        ``task_suite_name``; should be left as an empty string unless
        manual override is needed.

    Environment Parameters
    ~~~~~~~~~~~~~~~~~~~~~~
    env_img_res : int, default ``256``
        MuJoCo camera render resolution in pixels (height = width).
        Both ``camera_heights`` and ``camera_widths`` are set to this value.
    num_steps_wait : int, default ``10``
        Number of dummy (zero) actions executed at the start of each episode
        before any embedding extraction or action prediction. Allows gravity
        and soft-body contacts to settle to a stable configuration.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained_checkpoint: str = ""       # Required: path or HuggingFace Hub ID
    model_family: str = "openvla"         # Currently only "openvla" is supported

    # ── Action head ───────────────────────────────────────────────────────────
    use_l1_regression: bool = True        # Enable L1 continuous action head
    use_diffusion: bool = False           # Enable DDIM diffusion action head
    num_diffusion_steps: int = 50         # DDIM denoising iterations per prediction
    use_film: bool = False                # Enable FiLM language-visual modulation
    num_images_in_input: int = 2          # 2 = third-person + wrist; 1 = third-person only
    use_proprio: bool = True              # Include 8-dim proprio state as input

    # ── Inference ─────────────────────────────────────────────────────────────
    center_crop: bool = True              # Deterministic centre crop at inference time
    num_open_loop_steps: int = 8          # Actions executed per model query (chunk size)

    # ── Quantisation ──────────────────────────────────────────────────────────
    load_in_8bit: bool = False            # INT8 quantisation (bitsandbytes), mutually exclusive
    load_in_4bit: bool = False            # NF4 quantisation (bitsandbytes), mutually exclusive

    # ── Task ──────────────────────────────────────────────────────────────────
    task_suite_name: str = "libero_goal"  # LIBERO suite for env loading and key resolution
    unnorm_key: str = ""                  # Action un-normalisation key; auto-resolved

    # ── Environment ───────────────────────────────────────────────────────────
    env_img_res: int = 256                # MuJoCo render resolution (pixels, H = W)
    num_steps_wait: int = 10              # Warm-up steps before embedding extraction


# =============================================================================
# MODEL AND COMPONENT LOADER
# =============================================================================

def load_model_and_components(cfg: EmbeddingConfig) -> tuple:
    """
    Instantiate all neural network components needed for rollout inference
    and embedding extraction from an OpenVLA checkpoint.

    This function centralises all model-loading logic so that the main
    extraction loop in ``extract_embeddings_rollout()`` receives fully
    initialised, GPU-resident components ready for inference.

    Components are loaded conditionally based on architecture flags in ``cfg``,
    minimising GPU memory consumption when optional modules (diffusion head,
    proprio projector) are disabled. The action un-normalisation key is
    resolved and written back into ``cfg.unnorm_key`` so that ``get_vla_action``
    can correctly de-standardise predictions without further configuration.

    Loading Sequence
    ----------------
    1. **VLA backbone**: visual encoder + multimodal projector + causal LLM.
       This is the largest component and always loaded first.
    2. **Un-normalisation key**: walks through key name variants in
       ``model.norm_stats`` until a valid key is found, then writes it to
       ``cfg.unnorm_key``.
    3. **Processor**: HuggingFace processor combining the text tokenizer and
       the image pre-processing pipeline (resize, normalise, to-tensor).
    4. **Proprio projector** (optional): a small MLP mapping 8-dim state →
       LLM hidden dim; loaded only when ``cfg.use_proprio=True``.
    5. **Action head** (optional, graceful fallback): the continuous action
       prediction head. If the weight file is absent, the function assumes the
       head is fused into the LoRA checkpoint and continues with ``None``.
    6. **Noisy action projector** (optional): DDIM noise-step conditioning
       module; loaded only when ``cfg.use_diffusion=True``.
    7. **Resize size**: derives the ``(H, W)`` tuple expected by the image
       pre-processor from the config, to be passed to ``resize_image_for_policy``.

    Parameters
    ----------
    cfg : EmbeddingConfig
        Configuration object. ``cfg.unnorm_key`` is mutated in place with the
        resolved key string after this function returns.

    Returns
    -------
    model : torch.nn.Module
        Full VLA backbone in evaluation mode, on CUDA. Exposes
        ``model.vision_backbone``, ``model.projector``, ``model.language_model``,
        ``model.llm_dim``, ``model.norm_stats``, and ``model.device``.
    processor : transformers.ProcessorMixin
        Combined tokenizer and image pre-processor for the OpenVLA model.
    action_head : torch.nn.Module or None
        Loaded continuous action head, or ``None`` if not applicable or not
        found as a standalone file.
    proprio_projector : torch.nn.Module or None
        Proprioceptive state projector, or ``None`` if ``cfg.use_proprio=False``.
    noisy_action_projector : torch.nn.Module or None
        DDIM noise projector, or ``None`` if ``cfg.use_diffusion=False``.
    resize_size : tuple of (int, int)
        Target ``(H, W)`` resolution for policy input images.

    Raises
    ------
    AssertionError
        If neither ``cfg.task_suite_name`` nor its ``_no_noops`` variant
        exists in ``model.norm_stats``, indicating an incompatible checkpoint.
    """
    print(f"Loading model from: {cfg.pretrained_checkpoint}")

    # ── 1. VLA backbone ───────────────────────────────────────────────────────
    # Loads the vision encoder + projector + LLM from cfg.pretrained_checkpoint.
    # Respects cfg.load_in_8bit / cfg.load_in_4bit for quantised loading.
    model = get_vla(cfg)

    # ── 2. Action un-normalisation key resolution ─────────────────────────────
    # The model's norm_stats dict stores per-dimension (mean, std) statistics
    # for action de-standardisation, keyed by the dataset name.
    # Start with the task suite name as the primary candidate key.
    unnorm_key = cfg.task_suite_name   # e.g. "libero_goal"

    # If the plain key is absent, try the "_no_noops" variant that some
    # checkpoints use to denote datasets with no-op actions filtered out.
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"   # Fallback: e.g. "libero_goal_no_noops"

    # Fail with a descriptive message if no valid key was found.
    assert unnorm_key in model.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in VLA norm_stats!"
    )

    # Persist the resolved key so get_vla_action can find it on the config object.
    cfg.unnorm_key = unnorm_key

    # ── 3. Input processor ────────────────────────────────────────────────────
    # Returns a HuggingFace AutoProcessor combining the LLaVA tokenizer and
    # the SigLIP/CLIP image pre-processor (resize, normalise, to-tensor).
    processor = get_processor(cfg)

    # ── 4. Proprioceptive projector (optional) ────────────────────────────────
    proprio_projector = None          # Default: disabled
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,   # Output dim must match LLM hidden dimension exactly
            proprio_dim=8,   # Input: 3 (XYZ) + 3 (axis-angle) + 2 (gripper) = 8
        )

    # ── 5. Continuous action head (optional, graceful fallback) ───────────────
    action_head = None   # Default: disabled or fused
    if cfg.use_l1_regression or cfg.use_diffusion:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            print("✓ Action head loaded")
        except (AssertionError, FileNotFoundError):
            # Weight file not found as a standalone file — the head is assumed to
            # be fused into the LoRA adapter weights within the main checkpoint.
            print("⚠️  Action head not found, assuming integrated in model")
            # action_head remains None; get_vla_action handles this case gracefully

    # ── 6. DDIM noise projector (optional) ───────────────────────────────────
    noisy_action_projector = None   # Default: disabled
    if cfg.use_diffusion:
        # Projects the DDIM denoising timestep conditioning signal into the
        # LLM hidden space to condition the diffusion process.
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # ── 7. Image resize resolution ────────────────────────────────────────────
    # Derives the (H, W) tuple that the policy expects as input.
    # Typically (224, 224) or (256, 256) depending on the vision backbone.
    resize_size = get_image_resize_size(cfg)

    return (
        model,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        resize_size,
    )


# =============================================================================
# BDDL FILE PATH BUILDER
# =============================================================================

def build_bddl_path(task, level: str) -> str:
    """
    Construct the absolute filesystem path to the BDDL task definition file
    for a given task object and command variation level.

    LIBERO task definitions are stored as BDDL (Behavior Domain Definition
    Language) files under a per-suite subdirectory inside the LIBERO data
    directory. The default instruction uses the canonical BDDL file registered
    with the task object. Paraphrase variants are stored as separate files
    following a ``_syn_{level}`` naming convention appended to the base name.

    File Naming Convention
    ----------------------
    - **Default**: ``{bddl_root}/{problem_folder}/{bddl_file}``
      Example: ``.../bddl_files/libero_goal/pick_up_the_plate.bddl``
    - **Variant**: ``{bddl_root}/{problem_folder}/{base}_syn_{level}.bddl``
      Example: ``.../bddl_files/libero_goal/pick_up_the_plate_syn_l1.bddl``

    This function does **not** verify that the constructed path exists on disk.
    Callers must check existence with ``os.path.exists()`` before attempting
    to open the file.

    Parameters
    ----------
    task : libero.libero.benchmark.Task
        Task NamedTuple from the LIBERO benchmark registry. Must expose:
        - ``task.problem_folder`` : suite subdirectory name (e.g. ``"libero_goal"``)
        - ``task.bddl_file``      : canonical BDDL filename (e.g. ``"pick_up_the_plate.bddl"``)
    level : str
        Command variation level. One of:
        - ``"default"``  — canonical instruction from the registered BDDL file.
        - ``"l1"``       — minor synonym substitutions (low lexical divergence).
        - ``"l2"``       — structural rephrasing (medium lexical divergence).
        - ``"l3"``       — complete reformulation (high lexical divergence).

    Returns
    -------
    str
        Absolute path string to the BDDL file. May not exist on disk if the
        variant was not generated for this task.

    Examples
    --------
    >>> build_bddl_path(task, "default")
    '/data/libero/bddl_files/libero_goal/pick_up_the_plate.bddl'
    >>> build_bddl_path(task, "l1")
    '/data/libero/bddl_files/libero_goal/pick_up_the_plate_syn_l1.bddl'
    >>> build_bddl_path(task, "l3")
    '/data/libero/bddl_files/libero_goal/pick_up_the_plate_syn_l3.bddl'
    """
    if level == "default":
        # Use the canonical BDDL file registered with the task object.
        # get_libero_path("bddl_files") returns the absolute root of the LIBERO
        # BDDL files directory, independent of the current working directory.
        return os.path.join(
            get_libero_path("bddl_files"),   # e.g. "/data/libero/bddl_files"
            task.problem_folder,              # e.g. "libero_goal"
            task.bddl_file,                   # e.g. "pick_up_the_plate.bddl"
        )

    # Strip the ".bddl" extension from the canonical filename to get the base stem.
    # e.g. "pick_up_the_plate.bddl" → "pick_up_the_plate"
    base_name = task.bddl_file.replace(".bddl", "")

    # Append the variant suffix to construct the paraphrase BDDL filename.
    # e.g. "pick_up_the_plate" + "_syn_l1" + ".bddl" → "pick_up_the_plate_syn_l1.bddl"
    bddl_filename = f"{base_name}_syn_{level}.bddl"

    return os.path.join(
        get_libero_path("bddl_files"),   # LIBERO bddl_files root directory
        task.problem_folder,              # Suite subdirectory
        bddl_filename,                    # Constructed variant filename
    )


# =============================================================================
# OPENVLA PROMPT BUILDER
# =============================================================================

def build_prompt(task_label: str) -> str:
    """
    Format a raw task instruction string into the OpenVLA prompt template.

    OpenVLA was fine-tuned on data where each input was formatted as a
    question-answer pair using a specific template. Reproducing this template
    **exactly** at inference time is critical: even small deviations (e.g.
    missing newline, different casing) can shift the token distribution and
    degrade action quality, because the model's weights encode the template
    implicitly through the training objective.

    Template
    --------
    ``In: What action should the robot take to {instruction}?\\nOut:``

    The ``\\nOut:`` suffix marks the boundary of the action token generation
    region under causal attention, signalling to the LLM that the response
    (action tokens) should begin immediately after this position.

    Case Normalisation
    ------------------
    The instruction is lower-cased before insertion (``task_label.lower()``)
    to match the normalisation applied during the OpenVLA training data
    pipeline, where all task labels were stored in lower case.

    Parameters
    ----------
    task_label : str
        Raw instruction string, typically extracted from a BDDL ``:language``
        field (e.g. ``"Pick up the cream cheese"``).

    Returns
    -------
    str
        Fully formatted OpenVLA prompt string ready to be passed to
        ``processor(prompt, image)`` as the text input.

    Examples
    --------
    >>> build_prompt("Pick up the cream cheese")
    'In: What action should the robot take to pick up the cream cheese?\\nOut:'
    >>> build_prompt("OPEN THE DRAWER AND PUT THE BOWL INSIDE")
    'In: What action should the robot take to open the drawer and put the bowl inside?\\nOut:'
    """
    # Lower-case the instruction to match training data normalisation,
    # then interpolate into the fixed OpenVLA prompt template.
    # The f-string embeds a literal newline (\n) before "Out:" as required.
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


# =============================================================================
# MULTIMODAL HIDDEN-STATE EMBEDDING EXTRACTOR
# =============================================================================

def extract_embedding_from_forward(
    model,
    processor,
    prompt: str,
    primary_image: Image.Image,
    wrist_image: Optional[Image.Image] = None,
) -> np.ndarray:
    """
    Extract a text-conditioned, visually-grounded embedding from the OpenVLA
    model by manually executing a multimodal forward pass with hidden state
    extraction enabled.

    Motivation
    ----------
    The standard OpenVLA ``predict_action()`` API (and the underlying HuggingFace
    ``generate()`` call) does not return per-step hidden states. This function
    re-implements the internal multimodal forward pass step by step, making the
    language model's last-layer hidden states accessible without modifying the
    model's source code or relying on hooks.

    The resulting embedding captures the model's internal representation of
    "what the robot should be doing, given the current visual state and
    instruction" — a richer signal than a text-only embedding because it
    incorporates visual context through cross-attention.

    Multimodal Sequence Construction
    ---------------------------------
    OpenVLA interleaves visual and text tokens in a specific order that must be
    reproduced exactly to match the training distribution:

    .. code-block::

        Position:   [ 0  ] [ 1 … N ] [ N+1 … N+L-1 ]
        Content:    [ BOS ] [patches] [  text tokens  ]
        Attention:  [  1  ] [   1   ] [      mask     ]

    where ``N = n_patches`` (number of visual patch embeddings from the vision
    backbone) and ``L`` is the total number of text tokens including BOS.

    After the forward pass, hidden states at positions ``0`` (BOS) and
    ``N+1 … N+L-1`` (text tokens) are extracted. Visual patch positions
    ``1 … N`` are excluded because they carry low-level spatial signals that
    vary rapidly from step to step, while text-position states accumulate a
    holistic scene-conditioned task representation.

    Mean Pooling
    ------------
    The ``L`` text-position hidden state vectors (each of shape ``(hidden_dim,)``)
    are combined via attention-weighted mean pooling:

    .. math::

        \\mathbf{e} = \\frac{\\sum_{i=1}^{L} m_i \\, \\mathbf{h}_i}{\\sum_{i=1}^{L} m_i}

    where ``m_i \\in \\{0, 1\\}`` is the attention mask for token ``i`` and
    ``\\mathbf{h}_i`` is the corresponding last-layer hidden state. This zeros
    out any padding positions that were added by the tokenizer.

    Multi-Camera Input Handling
    ---------------------------
    When a ``wrist_image`` is provided (``num_images_in_input > 1``), the
    wrist frame is processed independently by the processor, and its pixel
    tensor is concatenated to the primary pixel tensor along dimension 1
    (the channel/frame dimension):

    .. code-block::

        inputs["pixel_values"] = cat([primary_pixels, wrist_pixels], dim=1)

    This concatenation matches the multi-image stacking used during OpenVLA
    fine-tuning on LIBERO, where both camera views were always included.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded OpenVLA model in evaluation mode, on CUDA. Must expose:
        - ``model.vision_backbone``       : visual encoder (e.g. SigLIP)
        - ``model.projector``             : patch-to-LLM projection layer
        - ``model.language_model``        : causal language model (e.g. LLaMA-2)
        - ``model.device``                : target CUDA device
    processor : transformers.ProcessorMixin
        OpenVLA processor: tokenises the prompt string and pre-processes images
        into normalised tensors with the correct resolution and channel order.
    prompt : str
        Formatted OpenVLA prompt string returned by ``build_prompt()``.
    primary_image : PIL.Image.Image
        Third-person camera frame as an RGB PIL Image. Must be mode ``"RGB"``.
    wrist_image : PIL.Image.Image or None, optional
        Wrist camera frame as an RGB PIL Image. When ``None``, only the
        primary image contributes to the visual representation.

    Returns
    -------
    np.ndarray of shape ``(hidden_dim,)``
        Mean-pooled last-layer hidden state over text token positions,
        in float32 precision on CPU. ``hidden_dim`` equals the LLM's
        hidden dimension (e.g. 4096 for a 7B model).

    Notes
    -----
    - The entire function runs under ``torch.no_grad()`` to prevent gradient
      accumulation, which would otherwise consume GPU memory proportional
      to the sequence length times the number of layers.
    - Inputs are cast to ``torch.bfloat16`` to match the mixed-precision
      dtype of the loaded model weights.
    - The output is explicitly detached, moved to CPU, and cast to ``float32``
      before NumPy conversion, ensuring no GPU tensors are held in memory
      after the function returns.
    """
    with torch.no_grad():   # Disable autograd: no gradients needed, saves VRAM

        # ── Step 1: Process primary (third-person) camera frame ────────────────
        # processor(prompt, image) tokenises the text and applies image
        # pre-processing (resize to model input size, normalise pixel values,
        # convert to tensor). Returns a dict with:
        #   "input_ids"      : (1, L) — tokenised prompt
        #   "attention_mask" : (1, L) — 1 for real tokens, 0 for padding
        #   "pixel_values"   : (1, C, H, W) — normalised image tensor
        inputs = processor(prompt, primary_image).to(
            model.device,          # Move all tensors to the model's CUDA device
            dtype=torch.bfloat16,  # Cast to bfloat16 to match model weight dtype
        )

        # ── Step 2: Append wrist camera frame (multi-view mode) ────────────────
        if wrist_image is not None:
            # Process the wrist image with the same prompt to get its pixel tensor.
            # The text tokens are irrelevant here; we only need the pixel values.
            wrist_inputs = processor(prompt, wrist_image).to(
                model.device,
                dtype=torch.bfloat16,
            )
            # Concatenate primary and wrist pixel tensors along the channel/frame
            # dimension (dim=1), matching the multi-image stacking convention
            # used during OpenVLA LIBERO fine-tuning.
            # Shape: (1, C, H, W) + (1, C, H, W) → (1, 2C, H, W)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]],
                dim=1,   # Concatenate along the channel/frame axis
            )

        # ── Step 3: Visual encoding — patch embeddings from vision backbone ────
        # Extract the raw pixel tensor; shape depends on whether wrist was added.
        pixel_values = inputs["pixel_values"]   # (1, C or 2C, H, W)

        # Pass through the vision encoder (e.g. SigLIP ViT).
        # Returns patch embeddings of shape (1, n_patches, vision_dim).
        # Each patch corresponds to a local image region; n_patches is fixed
        # by the vision backbone's architecture (e.g. 576 for a 384px input).
        patch_embeddings = model.vision_backbone(pixel_values)

        # ── Step 4: Visual projection — patch embeddings → LLM space ──────────
        # A linear or 2-layer MLP projector maps each patch embedding from
        # vision_dim (e.g. 1152 for SigLIP-400M) to llm_dim (e.g. 4096),
        # so they can be directly concatenated with text token embeddings.
        # Output shape: (1, n_patches, llm_dim).
        projected_patches = model.projector(patch_embeddings)

        # Record the patch count for use in sequence assembly and mask building.
        n_patches = projected_patches.shape[1]   # e.g. 576

        # ── Step 5: Text token embeddings ─────────────────────────────────────
        # Retrieve the tokenised prompt token IDs: shape (1, L).
        input_ids = inputs["input_ids"]   # (1, L)

        # Look up the learned embedding vector for each token ID.
        # get_input_embeddings() returns the embedding layer (nn.Embedding).
        # Calling it on input_ids performs a table lookup.
        # Output shape: (1, L, llm_dim).
        text_embeds = model.language_model.get_input_embeddings()(input_ids)

        # ── Step 6: Assemble the multimodal input sequence ─────────────────────
        # OpenVLA's training format interleaves tokens as:
        #   [BOS] [patch_1 … patch_N] [text_1 … text_{L-1}]
        # This must be reproduced exactly to activate the correct attention patterns.

        # Split text embeddings into BOS (position 0) and the rest (positions 1…L-1).
        bos_embed  = text_embeds[:, :1, :]   # (1, 1, llm_dim) — BOS token embedding
        text_rest  = text_embeds[:, 1:, :]   # (1, L-1, llm_dim) — remaining text tokens

        # Concatenate: [BOS | patches | text_rest] along the sequence axis (dim=1).
        multimodal_embeds = torch.cat(
            [bos_embed, projected_patches, text_rest],
            dim=1,   # Sequence length axis
        )
        # Final shape: (1, 1 + n_patches + L-1, llm_dim) = (1, n_patches + L, llm_dim)

        # ── Step 7: Build the extended attention mask ──────────────────────────
        # The original text attention mask has shape (1, L): 1 for real tokens,
        # 0 for padding. We must extend it to cover the n_patches patch positions.

        attention_mask_text = inputs["attention_mask"]   # (1, L)

        # Create an all-ones mask for the visual patch positions (always attended).
        patch_mask = torch.ones(
            (attention_mask_text.shape[0], n_patches),   # (1, n_patches)
            dtype=attention_mask_text.dtype,              # Same dtype as text mask (int64)
            device=attention_mask_text.device,            # Same device (CUDA)
        )

        # Assemble: [BOS mask | patch masks | remaining text masks]
        # Mirrors the sequence structure built in Step 6.
        attention_mask_multimodal = torch.cat(
            [
                attention_mask_text[:, :1],   # BOS attention: (1, 1)
                patch_mask,                    # Patch attention: (1, n_patches)
                attention_mask_text[:, 1:],   # Text attention (excl. BOS): (1, L-1)
            ],
            dim=1,   # Sequence length axis
        )
        # Final shape: (1, 1 + n_patches + L-1) = (1, n_patches + L)

        # ── Step 8: Language model forward pass ───────────────────────────────
        # Feed the full multimodal sequence to the causal LLM.
        # output_hidden_states=True causes the model to return hidden states
        # from all transformer layers (including the embedding layer).
        # return_dict=True wraps outputs in a ModelOutput object for named access.
        lm_outputs = model.language_model(
            inputs_embeds=multimodal_embeds,              # (1, n_patches+L, llm_dim)
            attention_mask=attention_mask_multimodal,     # (1, n_patches+L)
            output_hidden_states=True,                    # Return all layer states
            return_dict=True,                             # Named output dict
        )

        # ── Step 9: Extract last-layer hidden states ───────────────────────────
        # lm_outputs.hidden_states is a tuple of (num_layers + 1) tensors,
        # each of shape (1, n_patches+L, llm_dim).
        # Index [0] = embedding layer output; index [-1] = last transformer layer.
        # We use the last layer for the richest contextualised representation.
        last_hidden = lm_outputs.hidden_states[-1]   # (1, n_patches+L, llm_dim)

        # ── Step 10: Isolate text-position hidden states ──────────────────────
        # Extract hidden states only at text token positions, excluding
        # the n_patches visual patch positions in the middle of the sequence.
        #
        # Sequence layout (positions):
        #   0         : BOS token
        #   1 … N     : visual patch tokens (N = n_patches) ← excluded
        #   N+1 … N+L-1 : text tokens (L-1 tokens)
        #
        # Keep: positions 0 (BOS) and positions N+1 onward (text).
        text_hidden = torch.cat(
            [
                last_hidden[:, :1, :],               # BOS: (1, 1, llm_dim)
                last_hidden[:, 1 + n_patches:, :],   # Text tokens: (1, L-1, llm_dim)
            ],
            dim=1,   # Reassemble along sequence axis
        )
        # Shape after cat: (1, L, llm_dim) — matches original text sequence length

        # ── Step 11: Attention-masked mean pooling ─────────────────────────────
        # Average only over real (non-padded) text token positions using the
        # original text attention mask.
        #
        # Expand mask from (1, L) to (1, L, 1) for broadcast multiplication.
        mask = attention_mask_text.unsqueeze(-1).to(text_hidden.dtype)   # (1, L, 1)

        # Zero out padding positions by element-wise multiplication.
        masked = text_hidden * mask   # (1, L, llm_dim) — padded positions = 0

        # Sum attended embeddings, then normalise by the count of attended tokens.
        # clamp(min=1) prevents division by zero for the (unlikely) edge case
        # where every token is masked.
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # pooled shape: (1, llm_dim)

        # ── Step 12: Return as a 1D CPU float32 NumPy array ────────────────────
        # squeeze(0) removes the batch dimension: (1, llm_dim) → (llm_dim,).
        # .detach() removes the tensor from the computation graph.
        # .cpu() moves it from CUDA to host memory.
        # .float() casts from bfloat16 to float32 for NumPy compatibility.
        # .numpy() converts to a standard NumPy ndarray.
        return pooled.squeeze(0).detach().cpu().float().numpy()
        # Return shape: (llm_dim,), dtype: float32


# =============================================================================
# OBSERVATION PREPARATION
# =============================================================================

def prepare_observation(obs: dict, resize_size: tuple) -> tuple:
    """
    Convert a raw LIBERO observation dictionary into the structured format
    expected by both the VLA action predictor and the embedding extractor.

    This function serves as a single extraction point to avoid calling
    ``get_libero_image`` and ``get_libero_wrist_image`` multiple times
    in the episode loop. It simultaneously produces:

    1. A structured ``observation`` dict for ``get_vla_action()``.
    2. The raw un-resized third-person frame for PIL conversion in the
       embedding extractor.
    3. The raw un-resized wrist frame for PIL conversion in the embedding
       extractor.

    Why Return Un-Resized Frames?
    -----------------------------
    ``resize_image_for_policy`` produces images sized for the VLA's input
    processor, which may apply its own internal resizing. The PIL images
    passed to ``extract_embedding_from_forward`` are processed by
    ``processor(prompt, image)``, which applies the processor's own
    pre-processing pipeline independently. Passing the original resolution
    images to the processor avoids double-resizing artefacts.

    Proprioceptive State Layout
    ---------------------------
    The 8-dimensional state vector encodes the complete end-effector pose
    and gripper configuration:

    +------------------+-------+-----------------------------------------+
    | Component        | Shape | Description                             |
    +==================+=======+=========================================+
    | eef_pos          | (3,)  | End-effector XYZ position in world frame|
    +------------------+-------+-----------------------------------------+
    | eef_axis_angle   | (3,)  | Axis-angle orientation (converted from  |
    |                  |       | quaternion via ``quat2axisangle``)      |
    +------------------+-------+-----------------------------------------+
    | gripper_qpos     | (2,)  | Left and right gripper joint positions  |
    +------------------+-------+-----------------------------------------+
    | **Total**        | **(8,)** |                                    |
    +------------------+-------+-----------------------------------------+

    Parameters
    ----------
    obs : dict
        Raw observation dictionary from ``env.step()`` or
        ``env.set_init_state()``. Expected keys:
        - ``"robot0_eef_pos"``    : np.ndarray (3,)
        - ``"robot0_eef_quat"``   : np.ndarray (4,) quaternion (qx, qy, qz, qw)
        - ``"robot0_gripper_qpos"``: np.ndarray (2,)
        - Camera image keys accessed internally by ``get_libero_image`` and
          ``get_libero_wrist_image``.
    resize_size : tuple of (int, int)
        Target ``(H, W)`` resolution for the VLA's input images, used only
        for the resized images in the ``observation`` dict.

    Returns
    -------
    observation : dict
        Structured observation with keys:
        - ``"full_image"``  : np.ndarray (H', W', 3) — resized third-person frame
        - ``"wrist_image"`` : np.ndarray (H', W', 3) — resized wrist frame
        - ``"state"``       : np.ndarray (8,) — proprioceptive state vector
    img : np.ndarray of shape (H_raw, W_raw, 3), dtype uint8
        Un-resized third-person RGB frame. Used in the embedding extractor
        after conversion to PIL with ``Image.fromarray(img).convert("RGB")``.
    wrist_img : np.ndarray of shape (H_raw, W_raw, 3), dtype uint8
        Un-resized wrist RGB frame. Used in the embedding extractor.
    """
    # ── Extract raw RGB frames from the observation dict ──────────────────────
    img       = get_libero_image(obs)        # Third-person camera: (H, W, 3) uint8
    wrist_img = get_libero_wrist_image(obs)  # Wrist-mounted camera: (H, W, 3) uint8

    # ── Resize frames to the policy's expected input resolution ───────────────
    img_resized       = resize_image_for_policy(img, resize_size)        # (H', W', 3)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)  # (H', W', 3)

    # ── Assemble the observation dict consumed by get_vla_action() ────────────
    observation = {
        "full_image":  img_resized,       # Resized third-person frame for VLA input
        "wrist_image": wrist_img_resized, # Resized wrist frame for VLA input
        "state": np.concatenate((
            obs["robot0_eef_pos"],                         # (3,) XYZ end-effector position
            quat2axisangle(obs["robot0_eef_quat"]),        # (3,) axis-angle orientation
            obs["robot0_gripper_qpos"],                    # (2,) left/right gripper joints
        )),  # Concatenated shape: (8,)
    }

    # Return the structured observation AND the original (un-resized) raw frames
    return observation, img, wrist_img


# =============================================================================
# ACTION POST-PROCESSING
# =============================================================================

def process_action(action: np.ndarray, model_family: str) -> np.ndarray:
    """
    Post-process a raw predicted action vector before sending it to the LIBERO
    simulator.

    The model's action prediction pipeline outputs actions in the training
    data's normalised coordinate system. Two sequential corrections are needed
    to make the action compatible with LIBERO's ``env.step()`` interface:

    Step 1 — Gripper normalisation (``normalize_gripper_action``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The action head predicts the gripper dimension in the continuous range
    ``[0, 1]`` (as trained). LIBERO's environment expects gripper commands
    in ``{-1, +1}`` (binary: close/open). With ``binarize=True``, the function
    remaps ``[0, 0.5)`` → ``-1`` (close) and ``[0.5, 1]`` → ``+1`` (open),
    eliminating ambiguity at intermediate values.

    Step 2 — Sign inversion (``invert_gripper_action``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    During OpenVLA training data collection, the gripper signal was inverted
    to unify conventions across heterogeneous datasets: ``0 = open`` (unusual
    convention), ``1 = close`` (unusual convention). After Step 1, the binary
    value is numerically correct as a LIBERO command, but the sign has been
    flipped relative to what the environment expects. ``invert_gripper_action``
    restores the physical meaning:
    - Model ``+1`` (close in training) → LIBERO ``-1`` (close)
    - Model ``-1`` (open in training)  → LIBERO ``+1`` (open)

    This correction is only applied for the ``"openvla"`` model family, since
    other model families may use a consistent gripper convention.

    Parameters
    ----------
    action : np.ndarray of shape ``(action_dim,)``
        Raw action vector from the model. Convention:
        - Indices ``0:3``  : delta end-effector XYZ translation
        - Indices ``3:6``  : delta end-effector RPY rotation
        - Index   ``6``    : gripper command (continuous, in ``[0, 1]``)
    model_family : str
        Model family string. Gripper sign inversion only applied when
        this equals ``"openvla"``.

    Returns
    -------
    action : np.ndarray of shape ``(action_dim,)``
        Post-processed action vector ready to be sent to ``env.step()`` via
        ``action.tolist()``.
    """
    # Step 1: Remap gripper from continuous [0, 1] to binary {-1, +1}.
    # binarize=True applies threshold at 0.5 and maps to {-1, +1}.
    action = normalize_gripper_action(action, binarize=True)

    # Step 2: Invert gripper sign to undo the OpenVLA training convention flip.
    # Only applies to the "openvla" model family; other families are unchanged.
    if model_family == "openvla":
        action = invert_gripper_action(action)   # Flips the gripper dimension sign

    return action   # Shape: (action_dim,), gripper is now in LIBERO's {-1, +1} convention


# =============================================================================
# FIRST-STEP EMBEDDING EXTRACTOR
# =============================================================================

def extract_first_step_embedding(
    cfg: EmbeddingConfig,
    env,
    prompt: str,
    model,
    processor,
    resize_size: tuple,
    initial_state=None,
) -> np.ndarray:
    """
    Extract a single embedding from the first observation of an episode,
    without executing any real policy actions (first_step_only mode).

    This function implements the lightweight extraction mode where the
    environment is reset, the physics warm-up runs, and then exactly one
    multimodal forward pass is executed to capture the embedding at
    observation ``t = num_steps_wait`` — the very first frame the policy
    would act upon. No action prediction or environment stepping beyond
    the warm-up occurs.

    Scientific Rationale
    --------------------
    By extracting the embedding before any robot motion, this mode measures
    the model's *prior* encoding of the task instruction given only the initial
    scene configuration. This answers the question: "Does the model already
    distinguish between instruction variants at time t=0, before seeing any
    consequence of its own actions?" It is useful for:

    - Studying the language tower's disambiguation capacity in isolation.
    - Rapid benchmarking of embedding geometry without the cost of full
      policy rollouts (~20× faster than full rollout mode).
    - Controlling for confounds introduced by behavioural differences
      (if two instruction variants lead to different robot trajectories,
      their full-rollout embeddings may differ due to visual changes
      rather than language encoding differences).

    Episode Flow
    ------------
    1. **Reset**: ``env.reset()`` reloads the MuJoCo scene to the task's
       default configuration.
    2. **Initial state**: ``env.set_init_state(initial_state)`` restores the
       pre-recorded object placement and robot joint configuration for this
       rollout index. If ``None``, the default post-reset observation is used.
    3. **Warm-up**: ``cfg.num_steps_wait`` zero-action steps are executed.
       Objects settle under gravity; soft-body contacts resolve. No embeddings
       are extracted during warm-up.
    4. **Observation preparation**: frames and proprio state are extracted and
       formatted. Raw frames are converted to PIL Images.
    5. **Embedding extraction**: one call to ``extract_embedding_from_forward``
       returns the mean-pooled text-position hidden state at this observation.

    Parameters
    ----------
    cfg : EmbeddingConfig
        Configuration providing:
        - ``cfg.num_steps_wait``     : number of warm-up steps
        - ``cfg.model_family``       : for dummy action generation
        - ``cfg.num_images_in_input``: whether to include wrist image
    env : libero.libero.envs.OffScreenRenderEnv
        Active LIBERO simulation environment for this task. Must already be
        instantiated; not closed by this function.
    prompt : str
        Formatted OpenVLA prompt from ``build_prompt()``.
    model : torch.nn.Module
        Loaded VLA backbone in evaluation mode.
    processor : transformers.ProcessorMixin
        OpenVLA tokenizer and image pre-processor.
    resize_size : tuple of (int, int)
        Target ``(H, W)`` resolution for the VLA's input images.
    initial_state : np.ndarray or None, optional
        Pre-recorded simulator state from the ``.pruned_init`` file for
        this rollout index. When ``None``, the environment's default
        post-reset state is used.

    Returns
    -------
    np.ndarray of shape ``(hidden_dim,)``
        Single embedding extracted from the first actionable observation.
        Float32 precision, on CPU.
    """
    # ── Phase 1: Environment reset ─────────────────────────────────────────────
    env.reset()   # Reload MuJoCo scene; returns to task's canonical initial config

    # ── Phase 2: Restore pre-recorded initial state ────────────────────────────
    if initial_state is not None:
        # Apply the pre-recorded joint positions and object placements.
        # env.set_init_state() sets the simulator state and returns the
        # corresponding observation dictionary.
        obs = env.set_init_state(initial_state)
    else:
        # No initial state provided: use the environment's own post-reset obs.
        obs = env.get_observation()

    # ── Phase 3: Physics warm-up ───────────────────────────────────────────────
    # Execute cfg.num_steps_wait zero-action steps to allow gravity and
    # soft-body contact forces to settle before extracting any embedding.
    # This prevents the embedding from capturing an artefactual "mid-fall" scene.
    for _ in range(cfg.num_steps_wait):
        # get_libero_dummy_action returns a zero vector of the correct dimension
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        # We update obs each step so the final obs is the settled scene state

    # ── Phase 4: Prepare observation ──────────────────────────────────────────
    # Extract and format frames + proprio state from the settled observation.
    observation, img, wrist_img = prepare_observation(obs, resize_size)

    # ── Phase 5: Convert raw frames to PIL Images ─────────────────────────────
    # The OpenVLA processor expects PIL.Image.Image (not NumPy arrays).
    # Image.fromarray() converts uint8 (H, W, 3) → PIL Image.
    # .convert("RGB") ensures the mode is "RGB" even if the array had
    # an unexpected channel order.
    img_pil = Image.fromarray(img).convert("RGB")   # Third-person frame → PIL RGB

    # Only convert wrist image if the policy uses multi-image input (2 cameras).
    wrist_pil = (
        Image.fromarray(wrist_img).convert("RGB")
        if cfg.num_images_in_input > 1
        else None   # Single-camera mode: wrist image not used
    )

    # ── Phase 6: Extract and return the embedding ──────────────────────────────
    # One full multimodal forward pass; returns shape (hidden_dim,).
    embedding = extract_embedding_from_forward(
        model,      # VLA backbone
        processor,  # Tokenizer + image pre-processor
        prompt,     # Formatted OpenVLA prompt string
        img_pil,    # Third-person PIL Image
        wrist_pil,  # Wrist PIL Image or None
    )

    return embedding   # Shape: (hidden_dim,), float32, CPU

# =============================================================================
# SINGLE EPISODE RUNNER WITH EMBEDDING EXTRACTION
# =============================================================================

def run_single_episode(
    cfg: EmbeddingConfig,
    env,
    task_description: str,
    prompt: str,
    model,
    processor,
    resize_size: tuple,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    max_steps: int = 300,
) -> tuple:
    """
    Execute a single evaluation episode in the LIBERO simulation with
    simultaneous real-action execution and per-step embedding extraction.

    This is the core function of the full-rollout extraction mode. Unlike
    ``extract_first_step_embedding()``, which records only the model's initial
    representation before any interaction, this function lets the policy run
    to completion (or until the step budget is exhausted) and records an
    embedding at **every** active time step.

    The dual-purpose forward pass
    ------------------------------
    At each active step, the function performs two separate forward passes
    through the model:

    1. **Embedding pass** (``extract_embedding_from_forward``):
       A full multimodal forward pass with ``output_hidden_states=True``.
       Returns the mean-pooled last-layer hidden state over text-token
       positions — a (hidden_dim,) vector capturing the model's holistic
       scene-and-task representation at this step.

    2. **Action pass** (``get_vla_action``):
       The standard OpenVLA inference call. Produces a chunk of
       ``cfg.num_open_loop_steps`` predicted actions. This pass is only
       executed when the action queue is empty (i.e., every
       ``cfg.num_open_loop_steps`` steps), amortising inference cost.

    Both passes operate under ``torch.no_grad()`` and on the same observation,
    so the embedding precisely reflects the visual context that drove the
    action decision.

    Open-Loop Action Chunking
    -------------------------
    Rather than querying the model at every step, the policy predicts a
    *chunk* of ``cfg.num_open_loop_steps`` actions in a single forward pass.
    These are stored in a FIFO ``deque`` and consumed one at a time on
    successive steps. The model is re-queried only when the deque is emptied.
    This amortises the significant inference latency of large VLA models
    (typically 300–500 ms per forward pass at 7B parameters) across multiple
    simulator steps (typically ~50 ms each), enabling near-real-time execution
    on a single GPU.

    Episode Timeline
    ----------------
    ::

        t=0  … t=num_steps_wait-1  : warm-up phase — dummy actions, no embedding
        t=num_steps_wait … t=max_steps+num_steps_wait-1 : active phase
            ├─ prepare_observation()            → resized image + proprio state
            ├─ extract_embedding_from_forward() → embedding recorded
            ├─ (if queue empty) get_vla_action() → fill action queue
            ├─ action_queue.popleft()           → dequeue next action
            ├─ process_action()                 → normalise + invert gripper
            └─ env.step(action)                 → advance simulator one step
                └─ if done → success=True, break

    Parameters
    ----------
    cfg : EmbeddingConfig
        Evaluation configuration. Key fields consumed:
        - ``cfg.num_steps_wait``    : number of warm-up dummy-action steps.
        - ``cfg.num_open_loop_steps``: action chunk size / queue capacity.
        - ``cfg.num_images_in_input``: whether to include the wrist camera.
        - ``cfg.model_family``      : controls gripper sign correction.
        - ``cfg.use_film``          : FiLM conditioning flag forwarded to
          ``get_vla_action``.
    env : libero.libero.envs.OffScreenRenderEnv
        Instantiated LIBERO MuJoCo environment for the current task.
        Must be freshly created (or reset externally) before calling this
        function. The function calls ``env.reset()`` internally.
    task_description : str
        Raw task instruction string (not prompt-formatted) extracted from the
        BDDL file. Passed directly to ``get_vla_action()`` as the conditioning
        text. This is **distinct** from ``prompt``, which is the fully-formatted
        OpenVLA template used for embedding extraction.
    prompt : str
        Fully-formatted OpenVLA prompt string from ``build_prompt()``.
        Used exclusively by ``extract_embedding_from_forward()`` for the
        embedding forward pass.
    model : torch.nn.Module
        Loaded OpenVLA model in evaluation mode, on CUDA.
    processor : transformers.ProcessorMixin
        OpenVLA tokenizer + image pre-processor.
    resize_size : tuple of (int, int)
        Target ``(H, W)`` image resolution for policy input images.
    action_head : torch.nn.Module or None, optional
        Continuous action prediction head. ``None`` if integrated in the
        LoRA checkpoint.
    proprio_projector : torch.nn.Module or None, optional
        Proprioceptive state projector MLP. ``None`` if
        ``cfg.use_proprio=False``.
    noisy_action_projector : torch.nn.Module or None, optional
        DDIM noise-step conditioning projector. ``None`` if
        ``cfg.use_diffusion=False``.
    initial_state : np.ndarray or object or None, optional
        Pre-recorded simulator state loaded from a ``.pruned_init`` file.
        When provided, ``env.set_init_state(initial_state)`` is called after
        ``env.reset()`` to reproduce a specific object placement. When
        ``None``, the default stochastic post-reset state is used.
    max_steps : int, default 300
        Maximum number of **active** simulator steps (warm-up excluded).
        Episodes that do not terminate with ``done=True`` before this limit
        are cut off and marked as failures. Should be set from
        ``TASK_MAX_STEPS[task_suite_name]`` in the outer loop.

    Returns
    -------
    embeddings : list of np.ndarray, each of shape ``(hidden_dim,)``
        Per-step embeddings, one entry per active simulator step at which
        the model was queried for an embedding. List length equals the
        number of active steps executed (0 if the environment reset failed).
        Empty if an unrecoverable exception occurred before any step.
    success : bool
        ``True`` if and only if the LIBERO environment signalled task
        completion (``done=True``) before ``max_steps`` active steps were
        consumed. ``False`` on timeout or exception.
    num_steps : int
        Number of active steps for which embeddings were collected, equal
        to ``len(embeddings)``. Used by the outer loop for step-count
        statistics and progress reporting.

    Notes
    -----
    - All exceptions inside the main loop are caught and printed but do
      **not** propagate. This prevents a single corrupted episode from
      aborting the entire evaluation run. The episode is recorded as a
      failure with however many embeddings were collected before the error.
    - The ``done`` variable returned by the warm-up ``env.step()`` calls
      is intentionally discarded (``_, done, _``) because it is not
      meaningful during the stabilisation phase.
    - The step counter ``t`` counts total simulator steps including warm-up.
      The loop guard ``t < max_steps + cfg.num_steps_wait`` ensures that
      exactly ``max_steps`` active steps are available after warm-up
      completes.
    """

    # ── Phase 1: Reset and optionally set initial state ───────────────────────

    # Reset the simulator to a clean initial configuration, unloading any
    # state from a previous episode.
    env.reset()

    if initial_state is not None:
        # Restore the pre-recorded object and robot configuration from the
        # .pruned_init file so that each rollout starts from a known,
        # reproducible scene state.
        obs = env.set_init_state(initial_state)
    else:
        # Use the default stochastic post-reset observation (object positions
        # may vary slightly due to physics initialisation noise).
        obs = env.get_observation()

    # ── Phase 2: Initialise action queue ──────────────────────────────────────

    # A FIFO deque with bounded capacity acts as the open-loop action buffer.
    # maxlen=cfg.num_open_loop_steps ensures the deque never holds more than
    # one predicted chunk; older entries are automatically discarded if
    # extend() overflows (which should not happen in normal operation).
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # ── Phase 3: Initialise accumulators ──────────────────────────────────────

    embeddings: list = []   # Accumulates one (hidden_dim,) array per active step
    t: int = 0              # Total step counter (warm-up + active)
    success: bool = False   # Updated to True only on env.step() returning done=True

    # ── Phase 4: Main episode loop ─────────────────────────────────────────────

    try:
        # Loop guard: t must not exceed the total budget (warm-up + active steps).
        while t < max_steps + cfg.num_steps_wait:

            # ── 4a: Warm-up phase ─────────────────────────────────────────────
            # During the first num_steps_wait steps, execute zero-action
            # commands to let gravity settle objects into stable contact poses.
            # No embedding is extracted and no policy query is made.
            if t < cfg.num_steps_wait:
                # get_libero_dummy_action returns a zero vector of the correct
                # dimensionality for the given model family.
                # The returned done signal is discarded during warm-up.
                obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1     # Increment before continue to avoid an infinite loop
                continue   # Skip embedding extraction and action prediction

            # ── 4b: Observation preparation ───────────────────────────────────
            # Convert the raw environment observation dict into the structured
            # format consumed by both extract_embedding_from_forward and
            # get_vla_action.
            # Returns:
            #   observation : dict with "full_image", "wrist_image", "state"
            #   img         : raw (H, W, 3) uint8 third-person frame (for PIL)
            #   wrist_img   : raw (H, W, 3) uint8 wrist frame (for PIL)
            observation, img, wrist_img = prepare_observation(obs, resize_size)

            # ── 4c: PIL conversion for embedding forward pass ─────────────────
            # The OpenVLA processor (HuggingFace) expects PIL.Image.Image
            # objects, not raw NumPy arrays. Converting to "RGB" mode ensures
            # a consistent 3-channel image regardless of the source format.
            img_pil = Image.fromarray(img).convert("RGB")

            # Include the wrist camera only when the model was trained with
            # two camera views (num_images_in_input > 1). When None is passed,
            # extract_embedding_from_forward uses only the primary image.
            wrist_pil = (
                Image.fromarray(wrist_img).convert("RGB")
                if cfg.num_images_in_input > 1
                else None
            )

            # ── 4d: Embedding extraction ──────────────────────────────────────
            # Perform a full multimodal forward pass through the VLA and
            # extract the mean-pooled last-layer hidden state at text positions.
            # This is the "dual" forward pass — the action forward pass happens
            # separately below via get_vla_action.
            embedding = extract_embedding_from_forward(
                model,      # VLA backbone (vision + projector + LLM)
                processor,  # Tokenizer + image pre-processor
                prompt,     # Formatted OpenVLA prompt string
                img_pil,    # Third-person camera frame as PIL Image
                wrist_pil,  # Wrist camera frame as PIL Image, or None
            )
            # Append the (hidden_dim,) float32 NumPy array to the episode list
            embeddings.append(embedding)

            # ── 4e: Action chunk prediction (lazy, on empty queue) ────────────
            # Re-query the model only when the action buffer runs dry.
            # This amortises the VLA inference cost (300-500ms) across
            # cfg.num_open_loop_steps simulator steps (~50ms each).
            if len(action_queue) == 0:
                # get_vla_action returns a (num_open_loop_steps, action_dim)
                # numpy array of predicted actions for the current observation.
                actions = get_vla_action(
                    cfg,
                    model,
                    processor,
                    observation,          # Structured observation with images + state
                    task_description,     # Raw instruction string (not prompt-formatted)
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                # Load all predicted actions into the FIFO queue.
                # extend() appends each row (action_dim,) individually.
                action_queue.extend(actions)

            # ── 4f: Action execution ──────────────────────────────────────────
            # Dequeue the oldest predicted action from the front of the buffer.
            action = action_queue.popleft()

            # Post-process: normalise gripper from [0,1] to {-1,+1} and
            # invert the sign convention introduced during OpenVLA training.
            action = process_action(action, cfg.model_family)

            # Step the simulator. .tolist() converts the NumPy array to a plain
            # Python list, which is what LIBERO's env.step() expects.
            obs, reward, done, info = env.step(action.tolist())

            # ── 4g: Termination check ─────────────────────────────────────────
            if done:
                # The LIBERO goal-condition checker returned True, meaning all
                # task predicates are satisfied → episode is a success.
                success = True
                break   # Exit the while loop immediately; do not step further

            # Increment the global step counter for the next iteration.
            t += 1

    except Exception as e:
        # Catch any unexpected error (simulator crash, NaN in tensors, OOM, etc.)
        # and print a diagnostic message. The function returns whatever embeddings
        # were collected before the error, and success remains False.
        print(f"      Episode error: {e}")

    # Return the per-step embedding list, the success flag, and the step count.
    # len(embeddings) == number of active steps at which embeddings were extracted.
    return embeddings, success, len(embeddings)


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_embeddings_rollout(
    checkpoint_path: str,
    task_suite_name: str = "libero_goal",
    command_levels: tuple = ("default", "l1", "l2", "l3"),
    output_dir: str = "/mnt/beegfs/a.cardamone7/outputs/embeddings",
    resolution: int = 256,
    seed: int = 0,
    num_rollouts_per_task: int = 10,
    first_step_only: bool = False,
) -> tuple:
    """
    Orchestrate multimodal embedding extraction across all tasks in a LIBERO
    benchmark suite and across multiple command paraphrase levels, using either
    full rollout episodes or single-step snapshots.

    High-Level Workflow
    -------------------
    For each of ``num_tasks`` tasks in the suite and each of ``len(command_levels)``
    paraphrase variants, the function:

    1. Resolves the BDDL file for the (task, level) combination and extracts
       the instruction string from it.
    2. Creates a fresh LIBERO environment seeded with ``seed + rollout_idx``
       for each of the ``num_rollouts_per_task`` rollout episodes.
    3. Depending on ``first_step_only``:

       **Full rollout mode** (``first_step_only=False``):
         Runs ``run_single_episode()`` to completion. At every active step,
         ``extract_embedding_from_forward()`` records a (hidden_dim,) embedding.
         These step-level embeddings are averaged to produce one embedding per
         rollout, then all rollout embeddings are averaged to produce the final
         task-level entry.

       **First-step mode** (``first_step_only=True``):
         Runs ``extract_first_step_embedding()`` to capture a single embedding
         from the first post-warm-up observation, without executing any real
         actions. This mode is ~20× faster and is used to probe the model's
         *prior* task representation before any visual feedback from its motion.

    4. Stores all raw and aggregated data under a string key
       ``"task_{N:02d}_{level}"`` in the ``all_embeddings`` dict.

    5. After iterating over all tasks and levels, serialises ``all_embeddings``
       to a ``.pkl`` file and returns both the dict and the file path.

    Output Dictionary Schema
    ------------------------
    ``all_embeddings[key]`` is a dict with:

    .. code-block:: python

        {
            "task_id":           int,         # 0-indexed task identifier
            "task_name":         str,         # Task name from the benchmark
            "command_level":     str,         # "default", "l1", "l2", or "l3"
            "command_text":      str,         # Raw instruction from the BDDL file
            "prompt":            str,         # Formatted OpenVLA prompt string
            "embedding":         np.ndarray,  # (hidden_dim,) mean of rollout embeddings
            "embedding_per_rollout": np.ndarray,  # (num_rollouts, hidden_dim)
            "num_rollouts":      int,         # Number of successful rollout embeddings
            "first_step_only":   bool,        # Extraction mode flag
            # ── Additional fields for full rollout mode only ──
            "embedding_all_steps": np.ndarray,  # (total_steps, hidden_dim) concatenated
            "rollout_successes": list[bool],    # Per-rollout success flags
            "num_successes":     int,           # Count of successful rollouts
            "total_steps":       int,           # Sum of steps across all rollouts
            "success_rate":      float,         # num_successes / num_rollouts
        }

    Output File Naming Convention
    ------------------------------
    ``{output_dir}/rollout_embeddings_{suite}_{levels}_{mode}_r{N}.pkl``

    where:
    - ``{suite}``   = ``task_suite_name`` (e.g. ``"libero_goal"``)
    - ``{levels}``  = ``"_".join(command_levels)`` (e.g. ``"default_l1_l2_l3"``)
    - ``{mode}``    = ``"first_step"`` or ``"full"``
    - ``{N}``       = ``num_rollouts_per_task``

    Example: ``rollout_embeddings_libero_goal_default_l1_l2_l3_full_r10.pkl``

    Error Handling Strategy
    -----------------------
    - **Missing BDDL file**: logged and skipped (``continue``); other levels
      for the same task proceed normally.
    - **Missing ``:language`` field**: logged and skipped.
    - **Environment creation failure**: logged per rollout; other rollouts
      for the same (task, level) continue.
    - **Episode error**: caught inside ``run_single_episode``; partial results
      (if any) are kept.
    - **Empty rollout_embeddings list**: no key is written to ``all_embeddings``
      for that (task, level) pair; a warning is printed.

    This strategy prioritises collection completeness: a single failing rollout
    never aborts the entire run, which would waste hours of cluster compute time.

    Parameters
    ----------
    checkpoint_path : str
        Absolute path to the fine-tuned OpenVLA checkpoint directory, or a
        HuggingFace Hub model ID string. Forwarded to ``EmbeddingConfig`` and
        then to ``get_vla()``.
    task_suite_name : str, default ``"libero_goal"``
        LIBERO benchmark suite to evaluate. Valid values are the keys of
        ``TASK_MAX_STEPS``: ``"libero_spatial"``, ``"libero_object"``,
        ``"libero_goal"``, ``"libero_10"``, ``"libero_90"``.
    command_levels : tuple of str, default ``("default", "l1", "l2", "l3")``
        Paraphrase variation levels to extract embeddings for. Each level
        corresponds to a ``_syn_{level}.bddl`` file per task. ``"default"``
        uses the canonical registered BDDL file without a suffix.
    output_dir : str
        Absolute directory path where the output ``.pkl`` file is written.
        Created automatically (including parents) if it does not exist.
    resolution : int, default ``256``
        MuJoCo render resolution in pixels (height = width). Passed to
        ``EmbeddingConfig.env_img_res`` and the environment factory.
    seed : int, default ``0``
        Base random seed. Each rollout ``i`` of each task is seeded with
        ``seed + i``, ensuring reproducible but diverse initial conditions
        across rollouts. The model's forward passes are deterministic at
        inference time (``torch.no_grad()``; no stochastic sampling).
    num_rollouts_per_task : int, default ``10``
        Number of independent rollout episodes per (task, level) combination.
        With 10 tasks and 4 levels, the total number of episodes is
        ``10 × 4 × num_rollouts_per_task``.
    first_step_only : bool, default ``False``
        When ``True``, switch to first-step extraction mode: only the initial
        observation embedding is collected per rollout, no actions are executed,
        and no success/step statistics are recorded. Reduces wall-clock time by
        a factor of roughly ``max_steps / 1``.

    Returns
    -------
    all_embeddings : dict[str, dict]
        Nested dictionary described in the "Output Dictionary Schema" section.
        Keys are strings of the form ``"task_{N:02d}_{level}"``.
    output_file : str
        Absolute path of the saved ``.pkl`` file on disk.

    Notes
    -----
    - The model and all its components are loaded **once** before the outer
      task loop. This amortises the ~60-second model loading time across
      all episodes, which is critical for running efficiently on a cluster
      with a limited allocation window.
    - Each environment is created fresh per rollout (not per task) to prevent
      accumulation of MuJoCo state across episodes. The ``env.close()`` call
      in the ``finally`` block ensures proper teardown even on exception.
    - Pre-recorded initial states are cycled with ``rollout_idx % len(initial_states)``
      when ``num_rollouts_per_task`` exceeds the number of available states,
      ensuring every rollout has a valid starting configuration.
    """

    # ── Step 1: Build configuration ────────────────────────────────────────────
    # Construct an EmbeddingConfig with the parameters provided by the caller.
    # Remaining fields retain their dataclass defaults (use_l1_regression=True,
    # use_proprio=True, num_open_loop_steps=8, etc.).
    cfg = EmbeddingConfig(
        pretrained_checkpoint=checkpoint_path,   # Required: model checkpoint path
        task_suite_name=task_suite_name,          # Used for unnorm_key resolution
        env_img_res=resolution,                   # MuJoCo render resolution
    )

    # ── Step 2: Load model (once, shared across all tasks and levels) ──────────
    # All returned objects are GPU-resident in evaluation mode.
    (
        model,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        resize_size,
    ) = load_model_and_components(cfg)

    # ── Step 3: Initialise LIBERO benchmark suite ──────────────────────────────
    # benchmark.get_benchmark_dict() returns a mapping from suite name strings
    # to benchmark class factories. Calling the factory (with ()) instantiates
    # the suite and triggers lazy loading of task metadata.
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()   # Instantiate the benchmark suite
    num_tasks = task_suite.n_tasks                   # Number of tasks in the suite (e.g. 10)

    # Look up the per-suite maximum active step budget; fall back to 300 if the
    # suite name is not in TASK_MAX_STEPS (e.g. custom or future suite names).
    max_steps = TASK_MAX_STEPS.get(task_suite_name, 300)

    # ── Step 4: Print extraction configuration banner ─────────────────────────
    mode_str = "FIRST STEP ONLY" if first_step_only else "FULL ROLLOUT"
    print(f"\n{'='*80}")
    print(f"EMBEDDING EXTRACTION - {mode_str}")
    print(f"{'='*80}")
    print(f"Task suite: {task_suite_name} ({num_tasks} tasks)")
    print(f"Command levels: {command_levels}")
    print(f"Rollouts per task: {num_rollouts_per_task}")
    if not first_step_only:
        # Step budget is only relevant in full rollout mode
        print(f"Max steps per episode: {max_steps}")
    print(f"Total episodes per level: {num_tasks * num_rollouts_per_task}")
    print(f"{'='*80}\n")

    # ── Step 5: Outer accumulator ─────────────────────────────────────────────
    # Top-level results dict: maps "task_{N:02d}_{level}" → per-task-level data.
    all_embeddings: dict = {}

    # ── Step 6: Task loop ─────────────────────────────────────────────────────
    for task_id in range(num_tasks):

        # Retrieve the Task NamedTuple for this index from the benchmark suite.
        task = task_suite.get_task(task_id)

        # task.name is the canonical task identifier string; fall back to str(task)
        # for benchmark implementations that do not expose the .name attribute.
        task_name = getattr(task, 'name', str(task))

        # Load all pre-recorded initial simulator states for this task.
        # These are typically stored as .pruned_init files; the benchmark suite
        # handles path resolution and deserialization internally.
        initial_states = task_suite.get_task_init_states(task_id)

        print("=" * 80)
        print(f"Task {task_id + 1}/{num_tasks}: {task_name}")
        print("=" * 80)

        # ── Step 6a: Command level loop ───────────────────────────────────────
        for level in command_levels:

            # Resolve the absolute path to the BDDL file for this (task, level) pair.
            bddl_file = build_bddl_path(task, level)

            # Skip silently if the variant BDDL file does not exist on disk.
            # This is expected for levels that were not generated for all tasks.
            if not os.path.exists(bddl_file):
                print(f"  {level.upper():8s}: BDDL file not found")
                continue   # Move to the next command level for this task

            # Extract the `:language "..."` instruction string from the BDDL file.
            command = extract_command_from_bddl(bddl_file)

            # Skip if the language field is missing or malformed.
            if command is None:
                print(f"  {level.upper():8s}: Could not extract command")
                continue

            print(f"\n  {level.upper():8s}: {command}")

            # Format the raw instruction into the OpenVLA prompt template.
            prompt = build_prompt(command)

            # ── Step 6b: Rollout-level accumulators ───────────────────────────
            # rollout_embeddings: one (hidden_dim,) entry per rollout
            #   (first-step embedding, or mean over all step embeddings).
            rollout_embeddings: list = []

            # rollout_all_embeddings: for full rollout mode, one (T_i, hidden_dim)
            #   array per rollout, where T_i is the episode length in steps.
            rollout_all_embeddings: list = []

            # rollout_successes: per-rollout Boolean success flags (full mode only).
            rollout_successes: list = []

            successes: int = 0     # Running count of successful rollouts
            total_steps: int = 0   # Running sum of active steps across rollouts

            # ── Step 6c: Rollout loop ─────────────────────────────────────────
            for rollout_idx in range(num_rollouts_per_task):

                # Create a fresh environment for this rollout.
                # change_command=True instructs get_libero_env to swap the
                # default instruction with the variant extracted from the
                # paraphrase BDDL file, so that the environment's internal
                # task description matches the variant.
                try:
                    env, task_description, _ = get_libero_env(
                        task,
                        change_command=(level != "default"),  # True for l1/l2/l3
                        command_level=level if level != "default" else None,
                        resolution=resolution,
                    )
                    # Seed with seed + rollout_idx so each rollout has a distinct
                    # but reproducible physics initialisation state.
                    env.seed(seed + rollout_idx)
                except Exception as e:
                    # Environment creation can fail due to missing mesh files,
                    # BDDL parsing errors, or MuJoCo licensing issues.
                    print(f"    Rollout {rollout_idx+1}: Failed to create env - {e}")
                    continue   # Skip this rollout; continue with the next

                # Select the initial state for this rollout.
                # Use modular indexing to cycle through available states when
                # num_rollouts_per_task > len(initial_states).
                init_state = initial_states[rollout_idx % len(initial_states)]

                try:
                    if first_step_only:
                        # ── First-step mode ───────────────────────────────────
                        # Extract a single embedding from the first post-warm-up
                        # observation, without running the full policy loop.
                        # No action is predicted or executed.
                        embedding = extract_first_step_embedding(
                            cfg=cfg,
                            env=env,
                            prompt=prompt,
                            model=model,
                            processor=processor,
                            resize_size=resize_size,
                            initial_state=init_state,
                        )
                        # Append the (hidden_dim,) array to the per-level list.
                        rollout_embeddings.append(embedding)
                        print(f"    Rollout {rollout_idx+1:2d}/{num_rollouts_per_task}: ✓ (1 step)")

                    else:
                        # ── Full rollout mode ─────────────────────────────────
                        # Execute the full episode, collecting one embedding per
                        # active step, predicting real actions, and recording success.
                        episode_embeddings, success, num_steps = run_single_episode(
                            cfg=cfg,
                            env=env,
                            task_description=command,   # Raw instruction (not formatted)
                            prompt=prompt,              # Formatted prompt for embedding pass
                            model=model,
                            processor=processor,
                            resize_size=resize_size,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            initial_state=init_state,
                            max_steps=max_steps,
                        )

                        if episode_embeddings:
                            # Stack the (T, hidden_dim) step-embedding matrix
                            # for this episode to enable both mean and raw access.
                            rollout_emb = np.stack(episode_embeddings, axis=0)  # (T, D)

                            # Compute the temporal mean embedding for this rollout.
                            # This single (hidden_dim,) vector summarises the entire
                            # trajectory's embedding trajectory.
                            rollout_mean = np.mean(rollout_emb, axis=0)   # (D,)

                            # Record the per-rollout mean and raw step embeddings.
                            rollout_embeddings.append(rollout_mean)
                            rollout_all_embeddings.append(rollout_emb)     # (T, D)
                            rollout_successes.append(success)

                        # Accumulate global counters for this (task, level) summary.
                        successes += int(success)     # 1 if success, 0 otherwise
                        total_steps += num_steps       # Number of active steps executed

                        status = "✓" if success else "✗"
                        print(
                            f"    Rollout {rollout_idx+1:2d}/{num_rollouts_per_task}: "
                            f"{status} ({num_steps} steps)"
                        )

                except Exception as e:
                    # Catch rollout-level errors (e.g. CUDA OOM mid-episode)
                    # and continue with the next rollout without aborting.
                    print(f"    Rollout {rollout_idx+1}: Error - {e}")

                finally:
                    # Always attempt to close the environment, even on exception.
                    # This releases MuJoCo resources and OpenGL render contexts,
                    # preventing file descriptor leaks over many rollouts.
                    try:
                        env.close()
                    except Exception:
                        pass   # Silently ignore teardown errors

            # ── Step 6d: Aggregate rollout embeddings ─────────────────────────
            if rollout_embeddings:
                # Stack all per-rollout embeddings: (num_rollouts, hidden_dim).
                rollout_embeddings_arr = np.stack(rollout_embeddings, axis=0)

                # Compute the mean embedding across all rollouts: (hidden_dim,).
                # This is the primary embedding stored in the output dict.
                mean_embedding = np.mean(rollout_embeddings_arr, axis=0)

                # Construct the result key: zero-padded task index + level name.
                # Example: "task_03_l2" for task_id=3, level="l2".
                key = f"task_{task_id:02d}_{level}"

                # ── Step 6e: Build result dict for this (task, level) entry ───
                all_embeddings[key] = {
                    "task_id":              task_id,
                    "task_name":            task_name,
                    "command_level":        level,
                    "command_text":         command,         # Raw BDDL instruction
                    "prompt":               prompt,          # Formatted OpenVLA prompt
                    "embedding":            mean_embedding,  # (D,) mean across rollouts
                    "embedding_per_rollout": rollout_embeddings_arr,  # (R, D)
                    "num_rollouts":         len(rollout_embeddings),  # Actual count
                    "first_step_only":      first_step_only, # Extraction mode flag
                }

                # Augment with full-rollout-specific fields.
                if not first_step_only and rollout_all_embeddings:
                    all_embeddings[key]["embedding_all_steps"] = np.concatenate(
                        rollout_all_embeddings, axis=0  # (sum(T_i), D) across rollouts
                    )
                    all_embeddings[key]["rollout_successes"] = rollout_successes  # list[bool]
                    all_embeddings[key]["num_successes"]     = successes          # int
                    all_embeddings[key]["total_steps"]       = total_steps        # int
                    all_embeddings[key]["success_rate"]      = (
                        successes / max(len(rollout_embeddings), 1)   # Avoid division by zero
                    )

                # Print per-task-level summary line.
                if first_step_only:
                    print(
                        f"    Summary: {len(rollout_embeddings)} embeddings, "
                        f"shape: {mean_embedding.shape}"
                    )
                else:
                    print(
                        f"    Summary: {successes}/{num_rollouts_per_task} success, "
                        f"{total_steps} total steps, "
                        f"embedding shape: {mean_embedding.shape}"
                    )
            else:
                # No embeddings were collected for this (task, level) pair —
                # typically indicates all rollouts failed at env creation.
                print(f"    No embeddings extracted for {level}")

        print()   # Blank line between tasks for readability

    # ── Step 7: Serialise results to disk ─────────────────────────────────────
    # Create the output directory (and any missing parents) if it does not exist.
    os.makedirs(output_dir, exist_ok=True)

    # Build a descriptive filename encoding the suite, levels, mode, and rollout count.
    mode_suffix = "first_step" if first_step_only else "full"
    output_file = os.path.join(
        output_dir,
        f"rollout_embeddings_"
        f"{task_suite_name}_"               # e.g. "libero_goal"
        f"{'_'.join(command_levels)}_"      # e.g. "default_l1_l2_l3"
        f"{mode_suffix}_"                   # "first_step" or "full"
        f"r{num_rollouts_per_task}.pkl",    # e.g. "r10"
    )

    # Serialise the entire all_embeddings dict as a binary pickle file.
    # Protocol default (highest available) is used for compact storage.
    with open(output_file, "wb") as f:
        pickle.dump(all_embeddings, f)

    # ── Step 8: Print final summary ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    if all_embeddings:
        first_key = next(iter(all_embeddings.keys()))   # Arbitrary first entry for shape info
        print(f"Total entries: {len(all_embeddings)}")
        print(f"Mean embedding shape: {all_embeddings[first_key]['embedding'].shape}")
        print(f"Mode: {'First step only' if first_step_only else 'Full rollout'}")

        # Print per-level average success rates for full rollout mode.
        if not first_step_only:
            for level in command_levels:
                # Collect all entries for this command level.
                level_data = [
                    v for k, v in all_embeddings.items()
                    if v['command_level'] == level
                ]
                if level_data:
                    # Average success_rate across all tasks at this level.
                    avg_sr = np.mean([d['success_rate'] for d in level_data])
                    print(f"  {level.upper()} avg success rate: {avg_sr:.2%}")

    print(f"\nOutput saved to: {output_file}")

    return all_embeddings, output_file


# =============================================================================
# COMMAND-LINE INTERFACE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    CLI entry point for ``extract_embeddings_rollout.py``.

    Parses command-line arguments and calls ``extract_embeddings_rollout()``
    with the resolved values. This block is executed only when the script is
    run directly (``python extract_embeddings_rollout.py --checkpoint ...``),
    not when it is imported as a module.

    All arguments have sensible defaults matching the values used for the
    primary LIBERO-Goal experiments; only ``--checkpoint`` is required.

    Typical Usage
    -------------
    Full rollout extraction on libero_goal, all 4 command levels, 10 rollouts:

    .. code-block:: bash

        python extract_embeddings_rollout.py \\
            --checkpoint /path/to/openvla-checkpoint \\
            --task_suite libero_goal \\
            --command_levels default l1 l2 l3 \\
            --num_rollouts 10 \\
            --output_dir /mnt/beegfs/a.cardamone7/outputs/embeddings

    First-step-only extraction (fast baseline):

    .. code-block:: bash

        python extract_embeddings_rollout.py \\
            --checkpoint /path/to/openvla-checkpoint \\
            --first_step_only

    Extract only default and l1 levels:

    .. code-block:: bash

        python extract_embeddings_rollout.py \\
            --checkpoint /path/to/openvla-checkpoint \\
            --command_levels default l1
    """
    import argparse

    # ── Argument parser definition ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "Extract multimodal embeddings from OpenVLA during real inference "
            "rollouts on LIBERO benchmark tasks. Supports full rollout mode "
            "(embeddings at every step) and first-step-only mode (single "
            "embedding per episode, no actions executed)."
        )
    )

    # Required: path to the fine-tuned OpenVLA checkpoint directory.
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,       # Script cannot run without this
        help=(
            "Absolute path to the fine-tuned OpenVLA model checkpoint "
            "directory, or a HuggingFace Hub model ID string. "
            "Example: /mnt/beegfs/a.cardamone7/checkpoints/openvla-oft-libero"
        ),
    )

    # LIBERO benchmark suite to evaluate. Controls which tasks are loaded
    # and which step budget is applied.
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
        help=(
            "LIBERO benchmark suite name. One of: libero_spatial, "
            "libero_object, libero_goal, libero_10, libero_90. "
            "Default: libero_goal."
        ),
    )

    # One or more command paraphrase levels to extract embeddings for.
    # nargs="+" accepts space-separated values (e.g. --command_levels default l1 l2 l3).
    parser.add_argument(
        "--command_levels",
        type=str,
        nargs="+",                          # Accept 1 or more space-separated values
        default=["default", "l1", "l2", "l3"],
        help=(
            "Command paraphrase levels to process. Space-separated list. "
            "Example: --command_levels default l1 l2 l3. "
            "Default: all four levels."
        ),
    )

    # Output directory for the serialised .pkl file.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/beegfs/a.cardamone7/outputs/embeddings",
        help=(
            "Absolute directory path where the output .pkl file is saved. "
            "Created automatically if it does not exist. "
            "Default: /mnt/beegfs/a.cardamone7/outputs/embeddings"
        ),
    )

    # MuJoCo render resolution; affects both image quality and VRAM usage.
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "MuJoCo camera render resolution in pixels (H = W). "
            "Must match the resolution used during policy fine-tuning. "
            "Default: 256."
        ),
    )

    # Base random seed for environment initialisation.
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base random seed. Each rollout i is seeded with seed + i for "
            "reproducible but diverse episode initialisation. Default: 0."
        ),
    )

    # Number of rollout episodes per (task, level) combination.
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help=(
            "Number of independent rollout episodes per (task, command level) "
            "combination. Total episodes = num_tasks × len(command_levels) × "
            "num_rollouts. Default: 10."
        ),
    )

    # Boolean flag: when present, switches to first-step-only extraction mode.
    # action="store_true" sets the value to True when the flag appears on the CLI
    # and leaves it False when the flag is absent.
    parser.add_argument(
        "--first_step_only",
        action="store_true",     # Presence of flag → True; absence → False
        help=(
            "When set, extract only the embedding from the first post-warm-up "
            "observation per rollout, without executing any actions. Runs "
            "approximately 20× faster than full rollout mode and is useful "
            "for probing the model's prior task representation."
        ),
    )

    # ── Parse and forward to the main extraction function ─────────────────────
    args = parser.parse_args()

    # Convert command_levels from list to tuple to match the function signature
    # type annotation (tuple), though both are accepted at runtime.
    extract_embeddings_rollout(
        checkpoint_path=args.checkpoint,
        task_suite_name=args.task_suite,
        command_levels=tuple(args.command_levels),  # Convert list → tuple
        output_dir=args.output_dir,
        resolution=args.resolution,
        seed=args.seed,
        num_rollouts_per_task=args.num_rollouts,
        first_step_only=args.first_step_only,
    )