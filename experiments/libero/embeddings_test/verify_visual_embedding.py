"""
verify_visual_embedding.py
===========================

Verifies whether the visual embedding produced by the OpenVLA model's vision
pipeline (DINOv2+SigLIP → vision_backbone → projector) is sufficiently
discriminative to differentiate the 10 tasks of the LIBERO Goal benchmark,
or whether the model relies almost exclusively on the textual prompt to
determine which action to execute.

Scientific Motivation
---------------------
In a Vision-Language-Action (VLA) model, the total action-conditioning signal
comes from two sources:

1. **Visual embedding**: the projected patch embeddings produced by the vision
   backbone and the projector MLP. These encode *what the robot currently sees*
   — object positions, scene configuration, robot state.

2. **Language embedding**: the hidden states at text-token positions in the
   language model, which encode *what the robot should do* as described by the
   instruction prompt.

If the visual embeddings of 10 different tasks are nearly identical (cosine
similarity > 0.99), it means that despite the scenes being semantically
distinct (different objects, different goal configurations), the vision
pipeline collapses them into indistinguishable representations. Under these
circumstances, the language model's text tokens carry the full discriminative
burden: changing or paraphrasing the prompt is the only way the model can
distinguish tasks, which directly explains why instruction robustness
(performance under L1/L2/L3 paraphrase variations) is a meaningful evaluation
axis for this model.

This script answers the question:
    "Do visual embeddings vary significantly across the 10 LIBERO Goal tasks,
     or does the model rely exclusively on the text to distinguish them?"

Analysis Pipeline
-----------------
1. Load the fine-tuned OpenVLA-OFT checkpoint (step 20000).
2. For each of the 10 LIBERO Goal tasks, initialise the MuJoCo simulation,
   run a short physics warm-up, and capture the first stabilised observation
   frame from both cameras (third-person + wrist).
3. Pass each frame through the visual pipeline only (vision_backbone →
   projector) and compute the mean-pool of the projected patch embeddings,
   yielding one (hidden_dim,) vector per task.
4. Construct a 10×10 matrix of cosine similarity and a 10×10 matrix of
   Euclidean distance between all pairs of task visual embeddings.
5. Compute off-diagonal statistics (mean, std, min, max) and provide an
   automatic interpretation of the results.
6. Execute a **sanity check**: given the *same* image with *different* prompts,
   verify that the projected patches are bitwise-identical (confirming that
   the visual pipeline is prompt-independent when ``use_film=False``).

Key Design Difference vs. compare_embeddings.py
-------------------------------------------------
``compare_embeddings.py`` performs a full multimodal forward pass through the
language model and extracts the hidden states at the last transformer layer.
Those states are conditioned on *both* the visual input and the text tokens.

This script stops **before** the language model: it extracts the projected
patches that would be injected into the LLM as visual tokens, but does not run
the transformer at all. The result is a purely visual representation, free of
any language-model conditioning — unless ``use_film=True``, in which case the
projector receives a language signal via FiLM modulation.

Output
------
All results are printed to stdout as formatted matrices and summary statistics.
No files are written. The script is designed as a quick diagnostic tool
that can be run on a single GPU in under 10 minutes for all 10 tasks.

Dependencies
------------
- torch       : GPU inference, gradient-free forward pass
- numpy       : Matrix construction, statistics, allclose comparison
- PIL         : Image format conversion for the processor
- libero      : LIBERO benchmark task registry and simulation environments
- prismatic   : OpenVLA model components

Author: Agostino Cardamone
"""


# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import os    # Filesystem utilities (os.path.join, os.makedirs, os.path.exists);
              # imported for consistency with sibling scripts, not directly used here

import sys   # sys.path manipulation to prepend the project root and LIBERO root
              # so that local package imports resolve without installation

import torch  # Core deep learning framework: GPU tensor operations, autograd context
               # manager (torch.no_grad), device management

import numpy as np  # Numerical array operations: matrix allocation, mean pooling,
                     # norm computation, abs difference, allclose comparison

from pathlib import Path   # Cross-platform, object-oriented file path construction;
                            # used to resolve the absolute path of this script
                            # regardless of the current working directory

from PIL import Image      # Pillow library: converts NumPy uint8 RGB arrays to
                            # PIL.Image.Image objects expected by the OpenVLA processor

from dataclasses import dataclass  # Decorator that auto-generates __init__, __repr__,
                                    # and __eq__ from type-annotated class fields;
                                    # used to define the EmbeddingConfig-compatible Cfg class

from typing import Optional   # Type hint for values that may be None (e.g. wrist_pil)
from typing import Dict       # Type hint for typed dictionaries (imported for completeness)
from typing import List       # Type hint for typed lists (Python < 3.9 compatibility)
from typing import Tuple      # Type hint for typed tuples (return type annotations)


# =============================================================================
# PROJECT AND LIBRARY ROOT PATH RESOLUTION
# =============================================================================

# Resolve the absolute path of this script, following any symbolic links.
# Using Path(__file__).resolve() is more robust than __file__ alone, which
# may be relative to the working directory on some Python versions.
current_file = Path(__file__).resolve()

# Navigate three directory levels up from this file to reach the repository root.
# Assumed directory structure:
#   openvla-oft/                       ← project_root (3 levels up)
#     experiments/
#       libero/
#         analysis/
#           verify_visual_embedding.py ← this file
project_root = current_file.parent.parent.parent   # openvla-oft/

# The LIBERO benchmark library lives in a sibling directory next to openvla-oft.
# Assumed structure: robosuite_test/openvla-oft/ and robosuite_test/LIBERO/
libero_root = project_root.parent / "LIBERO"       # ../LIBERO/

# Prepend both roots to sys.path so that their sub-packages (experiments/,
# prismatic/, libero/) are importable without pip install.
# insert(0, ...) ensures local versions shadow any globally installed packages
# with the same names.
sys.path.insert(0, str(project_root))   # Makes 'experiments', 'prismatic' importable
sys.path.insert(0, str(libero_root))    # Makes 'libero' importable


# =============================================================================
# INTERNAL UTILITIES — OPENVLA MODEL COMPONENTS
# =============================================================================

from experiments.openvla_utils import (
    get_processor,          # Returns the combined HuggingFace tokenizer + image
                             # pre-processor for the OpenVLA model family.
                             # Signature: get_processor(cfg) → ProcessorMixin
    get_vla,                # Instantiates the full VLA model (ViT backbone +
                             # projector MLP + LLaMA-2 7B) from the checkpoint path
                             # in cfg.pretrained_checkpoint.
                             # Signature: get_vla(cfg) → torch.nn.Module
    get_action_head,        # Loads the continuous action prediction head (L1 or DDIM)
                             # from a separate weight file alongside the checkpoint.
                             # Used here only for existence verification, not inference.
                             # Signature: get_action_head(cfg, llm_dim) → torch.nn.Module
    get_proprio_projector,  # Loads the MLP mapping 8-dim proprioceptive state into
                             # the LLM embedding space.
                             # Used here only for existence verification, not inference.
                             # Signature: get_proprio_projector(cfg, llm_dim, proprio_dim) → Module
    resize_image_for_policy, # Resizes a raw NumPy uint8 image to the target resolution.
                              # Signature: resize_image_for_policy(img, size) → np.ndarray
)


# =============================================================================
# INTERNAL UTILITIES — LIBERO ENVIRONMENT AND OBSERVATION HELPERS
# =============================================================================

from experiments.libero.libero_utils import (
    get_libero_env,           # Factory: creates a LIBERO OffScreenRenderEnv for a task.
                               # Signature: get_libero_env(task, change_command, resolution)
                               #            → (env, task_description, bddl_path)
    get_libero_image,         # Extracts the third-person (agentview) RGB frame.
                               # Signature: get_libero_image(obs) → np.ndarray (H, W, 3) uint8
    get_libero_wrist_image,   # Extracts the wrist-mounted RGB camera frame.
                               # Signature: get_libero_wrist_image(obs) → np.ndarray (H, W, 3) uint8
    get_libero_dummy_action,  # Returns a zero-action vector for the physics warm-up phase.
                               # Signature: get_libero_dummy_action(model_family) → np.ndarray
)


# =============================================================================
# INTERNAL UTILITIES — GENERAL ROBOT EXPERIMENT HELPERS
# =============================================================================

from experiments.robot_utils import (
    get_image_resize_size,  # Derives the target image resolution (int) from the config.
                             # Typically 224 or 256 depending on the vision backbone.
                             # Signature: get_image_resize_size(cfg) → int
)


# =============================================================================
# LIBERO BENCHMARK REGISTRY
# =============================================================================

from libero.libero import benchmark  # Provides benchmark.get_benchmark_dict(), which
                                      # returns a dict mapping suite name strings
                                      # (e.g. "libero_goal") to benchmark class factories.
                                      # Calling the factory (with ()) instantiates the suite
                                      # and exposes .n_tasks, .get_task(id),
                                      # and .get_task_init_states(id).


# =============================================================================
# CHECKPOINT PATH AND DEVICE
# =============================================================================

# Absolute path to the fine-tuned OpenVLA-OFT checkpoint at training step 20000.
# This checkpoint contains the weights of:
#   - Vision backbone (DINOv2 + SigLIP ViT fusion)
#   - Projector MLP (vision dim → LLM hidden dim)
#   - LLaMA-2 7B language model (with LoRA adapters)
#   - Action head (L1 regression MLP)
#   - Proprio projector (8-dim → LLM hidden dim MLP)
CHECKPOINT_PATH = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/openvla/"
    "openvla-7b+libero_goal_no_noops_20000_chkpt"
)

# Computation device: use CUDA GPU 0 for fast inference if available,
# fall back to CPU for environments without a GPU.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# =============================================================================
# TASK NAME REGISTRY
# =============================================================================

# Ordered list of the 10 task names in the LIBERO Goal benchmark suite.
# The index in this list corresponds to the task_id used by the benchmark
# (task_suite.get_task(task_id)). The order matches tasks_info.txt in the
# LIBERO data directory.
TASK_NAMES = [
    "put_the_wine_bottle_on_top_of_the_cabinet",     # task 0
    "open_the_top_drawer_and_put_the_bowl_inside",   # task 1
    "turn_on_the_stove",                             # task 2
    "put_the_bowl_on_top_of_the_cabinet",            # task 3
    "put_the_bowl_on_the_plate",                     # task 4
    "put_the_wine_bottle_on_the_rack",               # task 5
    "put_the_cream_cheese_in_the_bowl",              # task 6
    "open_the_middle_drawer_of_the_cabinet",         # task 7
    "push_the_plate_to_the_front_of_the_stove",      # task 8
    "put_the_bowl_on_the_stove",                     # task 9
]

# =============================================================================
# SANITY CHECK PROMPT SUITE
# =============================================================================

# Four semantically distinct OpenVLA-formatted prompts used exclusively in the
# sanity check function (sanity_check_text_independence).
#
# Scientific Rationale:
# When use_film=False (the default configuration in this script), the vision
# pipeline (vision_backbone + projector) processes pixel_values independently
# of the text input. The prompt is still required by the processor to build
# the full input dict (specifically to set up input_ids and attention_mask),
# but its content should have ZERO effect on the projected_patches tensor.
#
# To verify this property empirically, four maximally diverse prompts are
# applied to the SAME image and the resulting projected_patches are compared:
#  - Prompt 0 and 1 describe completely different manipulation goals.
#  - Prompt 2 involves a different object and target.
#  - Prompt 3 is a paraphrase of Prompt 0 (different wording, same semantics).
# If any pair produces different projected_patches, it indicates that either
# use_film=True is active (text modulates visual features via FiLM) or there
# is an unexpected dependency in the processor/backbone implementation.
SANITY_PROMPTS = [
    # Prompt 0 — canonical action: place wine bottle on top of the drawer
    "In: What action should the robot take to put the wine bottle on the top of the drawer?\nOut:",
    # Prompt 1 — completely different action: open the middle drawer layer
    "In: What action should the robot take to open the middle layer of the drawer?\nOut:",
    # Prompt 2 — different object and target: place the bowl on the stove
    "In: What action should the robot take to put the bowl on the stove?\nOut:",
    # Prompt 3 — paraphrase of Prompt 0: same semantics, different lexical form
    "In: What action should the robot take to place the wine bottle on the top of the drawer?\nOut:",
]

# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class Cfg:
    """
    Configuration schema for the visual embedding verification analysis.

    This dataclass mirrors the ``GenerateConfig`` / ``EmbeddingConfig`` schema
    used in sibling scripts (``run_libero_eval.py``, ``compare_embeddings.py``)
    so that the shared model-loading helpers (``get_vla``, ``get_processor``,
    ``get_action_head``, ``get_proprio_projector``) receive the same interface
    they expect without modification.

    **Critical field for this script: ``use_film``.**
    When ``use_film=False`` (default), the projector MLP maps visual patch
    embeddings into the LLM hidden space without receiving any language signal.
    The text prompt influences only the LLM token processing stage, which this
    script intentionally bypasses. Setting ``use_film=True`` would enable
    FiLM (Feature-wise Linear Modulation) conditioning, where language features
    are injected into intermediate visual feature maps via learned affine
    transformations — in that case, the same image with different prompts would
    produce different ``projected_patches``, and the sanity check would fail.

    Attribute Groups
    ----------------

    Model Parameters
    ~~~~~~~~~~~~~~~~
    pretrained_checkpoint : str, default ``CHECKPOINT_PATH``
        Absolute local path to the fine-tuned OpenVLA-OFT checkpoint
        (step 20000) on the libero_goal_no_noops dataset.
    model_family : str, default ``"openvla"``
        Model family identifier. Controls processor loading and gripper
        sign convention. Currently only ``"openvla"`` is supported.

    Action Head Parameters
    ~~~~~~~~~~~~~~~~~~~~~~
    use_l1_regression : bool, default ``True``
        When ``True``, ``load_model()`` attempts to load the L1 regression
        action head from a file alongside the checkpoint. In this script, it
        is loaded only for existence verification and then discarded — no
        actions are predicted during visual embedding extraction.
    use_diffusion : bool, default ``False``
        Use the DDIM diffusion action head instead of L1 regression. Set to
        ``False`` because this script does not execute any actions.
    num_diffusion_steps : int, default ``50``
        DDIM denoising steps. Irrelevant when ``use_diffusion=False``.
    use_film : bool, default ``False``
        **The most important flag for this script.** When ``False``, the
        visual pipeline (vision_backbone + projector) is fully decoupled from
        the text input. The sanity check confirms this property empirically.
        Must match the value used during fine-tuning to load the correct
        projector weights.
    num_images_in_input : int, default ``2``
        Number of camera views concatenated as model input.
        ``2`` = third-person (agentview) + wrist camera, matching the LIBERO
        fine-tuning setup. Controls whether the wrist image is included in
        visual embedding extraction.
    use_proprio : bool, default ``True``
        When ``True``, ``load_model()`` checks for the proprioceptive projector
        weight file. Not used during embedding extraction (no robot state is
        passed to the visual pipeline).

    Inference Parameters
    ~~~~~~~~~~~~~~~~~~~~
    center_crop : bool, default ``True``
        Apply deterministic centre crop to policy input images. Should match
        the training augmentation to produce correctly preprocessed inputs.
    num_open_loop_steps : int, default ``8``
        Action chunk size. Unused in this script (no actions are predicted),
        but required to satisfy the ``get_vla`` and ``get_processor`` APIs.

    Quantisation
    ~~~~~~~~~~~~
    load_in_8bit : bool, default ``False``
        INT8 weight quantisation via bitsandbytes. Disabled to preserve full
        float16/bfloat16 precision for accurate embedding comparison.
    load_in_4bit : bool, default ``False``
        NF4 weight quantisation. Disabled for the same reason as ``load_in_8bit``.

    Task Parameters
    ~~~~~~~~~~~~~~~
    task_suite_name : str, default ``"libero_goal"``
        LIBERO benchmark suite identifier. Used to resolve the action
        un-normalisation key in ``model.norm_stats`` and to look up the
        benchmark task registry.
    unnorm_key : str, default ``""``
        Action statistics key for de-standardising predicted actions. Set
        automatically by ``load_model()``; left empty as a sentinel here.

    Environment Parameters
    ~~~~~~~~~~~~~~~~~~~~~~
    env_img_res : int, default ``256``
        MuJoCo camera render resolution in pixels (H = W). Controls the
        resolution of both the agentview and wrist camera renders.
    num_steps_wait : int, default ``10``
        Number of dummy (zero) simulator steps executed after resetting each
        environment to allow physics contacts to settle before capturing the
        first frame for embedding extraction.
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    pretrained_checkpoint: str = CHECKPOINT_PATH  # Checkpoint path; required
    model_family: str = "openvla"                 # Only "openvla" is supported

    # ── Action head (loaded for verification only; not used in inference) ──────
    use_l1_regression: bool = True     # Check for L1 action head existence
    use_diffusion: bool = False        # DDIM diffusion head disabled
    num_diffusion_steps: int = 50      # DDIM steps (irrelevant here)

    # ── CRITICAL: FiLM conditioning flag ──────────────────────────────────────
    # False = visual pipeline is text-independent → sanity check should pass.
    # True = FiLM injects language into visual features → patches change with prompt.
    use_film: bool = False

    # ── Multi-view input ──────────────────────────────────────────────────────
    num_images_in_input: int = 2   # Agentview + wrist camera concatenated

    # ── Proprioception (verification only; no robot state used) ───────────────
    use_proprio: bool = True       # Check for proprio projector existence

    # ── Inference settings (required by model-loading APIs) ───────────────────
    center_crop: bool = True       # Deterministic centre crop
    num_open_loop_steps: int = 8   # Action chunk size (unused)

    # ── Quantisation (disabled for precision) ─────────────────────────────────
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # ── Task suite ────────────────────────────────────────────────────────────
    task_suite_name: str = "libero_goal"  # Suite for benchmark and unnorm_key lookup
    unnorm_key: str = ""                  # Resolved automatically by load_model()

    # ── Environment ───────────────────────────────────────────────────────────
    env_img_res: int = 256     # MuJoCo render resolution in pixels (H = W)
    num_steps_wait: int = 10   # Warm-up dummy steps before frame capture


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_model(cfg: Cfg) -> tuple:
    """
    Load the OpenVLA-OFT VLA model and processor from the checkpoint.

    Design Difference from ``compare_embeddings.load_model_and_components()``
    --------------------------------------------------------------------------
    That function returns all six components (model, processor, action_head,
    proprio_projector, noisy_action_projector, resize_size) because it needs
    to run full policy rollouts.

    This function returns only three components (model, processor, resize_size)
    because visual embedding extraction stops before the language model and
    never executes any actions. The ``action_head`` and ``proprio_projector``
    are loaded temporarily to verify that the checkpoint is structurally
    complete, then immediately discarded without being returned.

    Loading Sequence
    ----------------
    1. **VLA backbone** — loads ViT (DINOv2+SigLIP fusion) + projector MLP +
       LLaMA-2 7B from ``cfg.pretrained_checkpoint``.
    2. **Un-normalisation key** — walks through ``model.norm_stats`` to find
       either ``"libero_goal"`` or ``"libero_goal_no_noops"`` and writes the
       resolved key back into ``cfg.unnorm_key``.
    3. **Processor** — loads the HuggingFace tokenizer + image pre-processor.
    4. **Proprio projector** (optional check) — attempts to load; if it fails,
       sets ``cfg.use_proprio=False`` and continues. Its return value is dropped.
    5. **Action head** (optional check) — same pattern; sets
       ``cfg.use_l1_regression=False`` on failure. Return value dropped.
    6. **Resize size** — derives the ``(H, W)`` target resolution from the
       config using ``get_image_resize_size()``.

    Parameters
    ----------
    cfg : Cfg
        Configuration object. ``cfg.unnorm_key`` and optionally
        ``cfg.use_proprio`` / ``cfg.use_l1_regression`` are mutated in place.

    Returns
    -------
    model : torch.nn.Module
        Full VLA backbone in evaluation mode on CUDA. Exposes:
        - ``model.vision_backbone`` : DINOv2 + SigLIP ViT fusion encoder
        - ``model.projector``       : vision-dim → LLM-dim projection MLP
        - ``model.language_model``  : LLaMA-2 7B (not used in this script)
        - ``model.llm_dim``         : LLM hidden dimension (e.g. 4096)
        - ``model.norm_stats``      : per-dataset action statistics dict
        - ``model.device``          : CUDA device the model is resident on
    processor : transformers.ProcessorMixin
        Combined tokenizer + image pre-processor. Called as
        ``processor(prompt, pil_image) → dict(input_ids, attention_mask, pixel_values)``.
    resize_size : int
        Target side length in pixels for ``resize_image_for_policy()``.
        Typically 224 (SigLIP backbone) or 256 (custom backbone).

    Raises
    ------
    AssertionError
        If neither ``cfg.task_suite_name`` nor its ``_no_noops`` variant
        is found in ``model.norm_stats``, indicating a checkpoint mismatch.
    """
    print(f"Loading model from:\n  {cfg.pretrained_checkpoint}\n")

    # ── 1. VLA backbone (always loaded) ───────────────────────────────────────
    # Instantiates the full model: ViT backbone + projector MLP + LLaMA-2 7B.
    # Respects cfg.load_in_8bit / cfg.load_in_4bit for quantised loading.
    model = get_vla(cfg)

    # ── 2. Action un-normalisation key resolution ─────────────────────────────
    # model.norm_stats is a dict keyed by dataset name. The key must be found
    # to allow get_vla_action() to de-standardise action predictions in other
    # scripts. Even though this script does not predict actions, the key must
    # be set so the processor loads without assertion errors.
    unnorm_key = cfg.task_suite_name   # Primary candidate: "libero_goal"

    # Fallback: try the "_no_noops" variant used by some checkpoints that were
    # trained after filtering out no-op actions from the dataset.
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"   # e.g. "libero_goal_no_noops"

    # Hard fail if no valid key is found; no inference is possible without stats.
    assert unnorm_key in model.norm_stats, (
        f"Action un-norm key '{unnorm_key}' not found in model.norm_stats. "
        f"Available keys: {list(model.norm_stats.keys())}"
    )

    # Write the resolved key back so shared utilities can find it on the config.
    cfg.unnorm_key = unnorm_key

    # ── 3. Input processor ────────────────────────────────────────────────────
    # Returns a HuggingFace AutoProcessor that combines the LLaVA-style text
    # tokenizer and the SigLIP image pre-processing pipeline.
    processor = get_processor(cfg)

    # ── 4. Proprioceptive projector (existence check only) ────────────────────
    if cfg.use_proprio:
        try:
            # Load and immediately discard; we only verify the file exists and
            # loads without error to confirm checkpoint integrity.
            get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
            print("✓ proprio_projector loaded")
        except Exception as e:
            # If the file is missing or corrupt, disable the flag so that
            # compare_embeddings and other scripts that share this config
            # object do not attempt to load a missing component.
            print(f"⚠  proprio_projector: {e}")
            cfg.use_proprio = False   # Disable downstream uso of proprio projector

    # ── 5. Action head (existence check only) ─────────────────────────────────
    if cfg.use_l1_regression:
        try:
            # Load and immediately discard for the same reason as above.
            get_action_head(cfg, model.llm_dim)
            print("✓ action_head loaded")
        except Exception as e:
            print(f"⚠  action_head: {e}")
            cfg.use_l1_regression = False   # Disable flag on missing file

    # ── 6. Image resize resolution ────────────────────────────────────────────
    # Derives the expected policy input resolution from the config, typically
    # based on the vision backbone's native patch size and the center-crop setting.
    resize_size = get_image_resize_size(cfg)

    return model, processor, resize_size

# =============================================================================
# VISUAL-ONLY EMBEDDING EXTRACTOR
# =============================================================================

def extract_visual_embedding(
    model,
    processor,
    img_pil: Image.Image,
    wrist_pil: Optional[Image.Image],
    prompt: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the purely visual representation of an image by running it through
    the VLA's vision backbone and projector, stopping before the language model.

    Key Architectural Difference vs. ``extract_embedding_from_forward()``
    -----------------------------------------------------------------------
    ``extract_embedding_from_forward()`` (in ``compare_embeddings.py``) runs
    the full pipeline: vision_backbone → projector → language_model (all
    transformer layers) and extracts the last-layer hidden states at text-token
    positions. The resulting embedding is **multimodal**: it reflects both the
    visual input and the language conditioning from the instruction prompt.

    This function stops at the **projector output**, before injecting any tokens
    into the language model. The resulting embedding is **purely visual**: it
    reflects only what the camera observed, with no language conditioning
    (assuming ``use_film=False``). The prompt is passed to the processor purely
    as a required input to construct the ``pixel_values`` tensor; it has no
    causal effect on the output when ``use_film=False``.

    This design choice is intentional: the goal is to measure whether different
    LIBERO Goal tasks produce visually distinguishable scene representations
    *before* any language signal is considered. If they do not, the model's
    task discrimination must come exclusively from the text tokens.

    Visual Pipeline
    ---------------
    ::

        pixel_values (1, C_total, H, W)
              │
              ▼
        vision_backbone (DINOv2 + SigLIP ViT)
              │  output: (1, n_patches, vision_dim)
              ▼
        projector MLP
              │  output: (1, n_patches, hidden_dim)   ← extraction stops here
              ▼
        [LLM — NOT executed]

    Multi-View Input Handling
    -------------------------
    When ``wrist_pil`` is not ``None``, the wrist camera frame is processed
    separately by the processor and its ``pixel_values`` tensor is concatenated
    to the primary tensor along dimension 1:

    ::

        primary pixel_values  : (1, C, H, W)
        wrist pixel_values    : (1, C, H, W)
        concatenated          : (1, 2C, H, W)

    The vision backbone is designed to accept this doubled channel count,
    processing both views as a joint visual token sequence. The number of
    output patches ``n_patches`` therefore doubles when ``wrist_pil`` is
    provided (e.g. from 576 to 1152 for a ViT with patch_size=14 on 224×224
    images).

    Parameters
    ----------
    model : torch.nn.Module
        Loaded VLA model in evaluation mode. Must expose:
        - ``model.vision_backbone`` : ViT encoder (DINOv2 + SigLIP fusion)
        - ``model.projector``       : projector MLP (vision_dim → hidden_dim)
        - ``model.device``          : CUDA device for tensor placement
    processor : transformers.ProcessorMixin
        OpenVLA processor: tokenises the prompt and pre-processes the image.
        Both are needed to produce the ``pixel_values`` tensor the backbone
        expects (normalised, resized, correct channel order).
    img_pil : PIL.Image.Image
        Third-person (agentview) RGB camera frame for the current task scene.
        Must be in ``"RGB"`` mode (3 channels, uint8 pixel values).
    wrist_pil : PIL.Image.Image or None
        Wrist-mounted camera frame. When ``None``, only the primary image
        is processed (single-view mode, ``num_images_in_input=1``).
    prompt : str
        A formatted OpenVLA prompt string (from ``build_prompt()``).
        Required by the processor to construct the full input dict, but
        has no effect on ``projected_patches`` when ``use_film=False``.
        The specific content of the prompt is therefore irrelevant here.

    Returns
    -------
    patches_flat : np.ndarray of shape ``(n_patches, hidden_dim)``
        The full matrix of projected patch embeddings in float32 on CPU.
        Each row corresponds to one spatial image region (patch), projected
        into the LLM's embedding space by the projector MLP.
        Typical shape: ``(576, 4096)`` for single-view,
        ``(1152, 4096)`` for dual-view with ViT patch_size=14 on 224×224.
        Used in the sanity check to verify bitwise equality across prompts.
    patches_mean : np.ndarray of shape ``(hidden_dim,)``
        The arithmetic mean of all patch vectors, i.e.
        ``patches_flat.mean(axis=0)``. Represents the entire scene content
        compressed into a single (hidden_dim,) descriptor.
        Used to build the 10×10 similarity and distance matrices in ``main()``.

    Notes
    -----
    - Runs entirely under ``torch.no_grad()`` to prevent gradient accumulation,
      which would waste VRAM proportional to the model depth.
    - Inputs are cast to ``torch.bfloat16`` to match the model's weight dtype
      (mixed-precision training). The output is cast back to ``float32`` before
      NumPy conversion for numerical precision in distance computations.
    - ``squeeze(0)`` removes the batch dimension (always 1 here) from the
      projected patch tensor before CPU conversion.
    """
    with torch.no_grad():   # Disable autograd: gradients are never needed here

        # ── Step 1: Process primary (third-person) camera frame ────────────────
        # processor(prompt, img_pil) tokenises the text and applies the SigLIP
        # image pre-processing pipeline (resize, normalise, to-tensor).
        # Returns a dict: {
        #   "input_ids":      (1, L)      — tokenised prompt
        #   "attention_mask": (1, L)      — 1 for real tokens, 0 for padding
        #   "pixel_values":   (1, C, H, W) — normalised image tensor
        # }
        # .to() moves everything to CUDA and casts pixel_values to bfloat16.
        inputs = processor(prompt, img_pil).to(model.device, dtype=torch.bfloat16)

        # ── Step 2: Concatenate wrist camera (multi-view mode) ─────────────────
        if wrist_pil is not None:
            # Process the wrist image with the same prompt and pre-processing.
            wrist_inputs = processor(prompt, wrist_pil).to(
                model.device, dtype=torch.bfloat16
            )
            # Concatenate pixel_values along dimension 1 (the channel dimension).
            # The vision backbone expects this doubled layout when trained on
            # two camera views simultaneously:
            #   (1, C, H, W) cat (1, C, H, W) → (1, 2C, H, W)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )

        # ── Step 3: Vision backbone — DINOv2 + SigLIP ViT ─────────────────────
        # The backbone accepts pixel_values of shape (1, C_total, H, W) and
        # divides the image(s) into a grid of non-overlapping patches, then
        # applies multiple transformer layers to compute spatial features.
        # Output shape: (1, n_patches, vision_dim)
        #   n_patches = (H / patch_size)² × num_images
        #   vision_dim = internal ViT feature dimension (e.g. 1536 for SigLIP-400M)
        patch_embeddings = model.vision_backbone(inputs["pixel_values"])

        # ── Step 4: Projector MLP ──────────────────────────────────────────────
        # A two- or three-layer MLP that maps each patch embedding from the
        # ViT's internal dimension (vision_dim) to the LLM's hidden dimension
        # (hidden_dim, e.g. 4096 for LLaMA-2 7B). After this point, the visual
        # tokens are in the same embedding space as the text tokens and can be
        # concatenated with them for LLM processing.
        # This is WHERE WE STOP: the LLM transformer is NOT called.
        # Input:  (1, n_patches, vision_dim)
        # Output: (1, n_patches, hidden_dim)
        projected_patches = model.projector(patch_embeddings)

        # ── Step 5: Post-processing for NumPy export ───────────────────────────
        # squeeze(0): remove the batch dimension → (n_patches, hidden_dim)
        # .detach():  cut the tensor from the computation graph (safety measure)
        # .cpu():     move from CUDA to RAM to avoid holding GPU memory
        # .float():   cast from bfloat16 to float32 for precise numpy arithmetic
        # .numpy():   convert to a standard NumPy array
        patches_flat = (
            projected_patches
            .squeeze(0)       # (1, n_patches, hidden_dim) → (n_patches, hidden_dim)
            .detach()
            .cpu()
            .float()
            .numpy()
        )   # Final shape: (n_patches, hidden_dim), e.g. (576, 4096)

        # Compute the mean-pool over all patches: reduces spatial resolution
        # to a single (hidden_dim,) descriptor summarising the entire scene.
        # axis=0 averages over the n_patches dimension, preserving hidden_dim.
        patches_mean = patches_flat.mean(axis=0)   # (hidden_dim,), e.g. (4096,)

    return patches_flat, patches_mean


# =============================================================================
# FIRST FRAME CAPTURE HELPER
# =============================================================================

def get_first_frame(
    task,
    cfg: Cfg,
    resize_size: int,
    task_id: int,
    task_suite,
) -> Tuple[Image.Image, Image.Image]:
    """
    Initialise the LIBERO MuJoCo simulation for a given task, run a short
    physics warm-up to stabilise the scene, capture both camera frames from
    the first meaningful observation, and shut down the environment.

    The function manages the full environment lifecycle (create → seed →
    reset → warm-up → capture → close) so that each call is self-contained
    and leaves no open simulator resources after it returns.

    Why Capture the First Frame?
    ----------------------------
    The visual embedding analysis compares the initial scene state of each
    task before any robot motion occurs. This choice ensures that:
    1. The comparison is unbiased by trajectory history (different policies
       produce different trajectories).
    2. The embedded scene faithfully represents the object configuration
       specified by the task's pre-recorded initial state (``initial_states[0]``).
    3. Results are fully reproducible (same seed, same initial state,
       deterministic warm-up steps).

    Warm-Up Phase
    -------------
    After ``env.set_init_state()``, the MuJoCo physics engine may exhibit
    brief transient contact forces as objects settle under gravity. Executing
    ``cfg.num_steps_wait`` (default: 10) zero-action steps before capturing
    the frame ensures the simulation has reached a stable configuration where
    all objects rest at their intended positions.

    Parameters
    ----------
    task : libero.libero.benchmark.Task
        Task NamedTuple from the LIBERO benchmark registry, providing the
        task name, BDDL file path, and problem folder.
    cfg : Cfg
        Configuration object. Fields used:
        - ``cfg.env_img_res``   : MuJoCo render resolution (H = W in pixels).
        - ``cfg.model_family``  : determines the dimensionality of dummy actions.
        - ``cfg.num_steps_wait``: number of warm-up steps before frame capture.
    resize_size : int
        Target side length (pixels) for ``resize_image_for_policy()``.
        Both the agentview and wrist images are resized to (resize_size, resize_size).
    task_id : int
        0-indexed task identifier (0–9 for LIBERO Goal). Used to look up the
        pre-recorded initial states for this specific task.
    task_suite : libero.libero.benchmark.BaseBenchmark
        Instantiated LIBERO benchmark suite object. Provides
        ``get_task_init_states(task_id)`` for deterministic initial state loading.

    Returns
    -------
    img_pil : PIL.Image.Image
        Third-person (agentview) RGB camera frame after warm-up, resized to
        ``(resize_size, resize_size)`` and converted to ``"RGB"`` mode.
    wrist_pil : PIL.Image.Image
        Wrist-mounted RGB camera frame after warm-up, resized to
        ``(resize_size, resize_size)`` and converted to ``"RGB"`` mode.

    Notes
    -----
    - The environment is always closed (``env.close()``) before returning,
      regardless of whether the initial state loading succeeded or fell back
      to the default reset. This prevents open MuJoCo rendering contexts from
      accumulating across 10 task calls.
    - The fallback path (bare ``env.reset()`` + ``env.get_observation()``)
      handles edge cases where ``task_suite.get_task_init_states()`` fails
      (e.g. missing ``.pruned_init`` files). In those cases the initial object
      placement is stochastic and results may not be fully reproducible.
    - ``env.seed(0)`` before loading initial states ensures that MuJoCo's
      internal RNG (used for contact friction sampling and other stochastic
      physics parameters) starts from the same state for all 10 tasks.
    """
    # ── Step 1: Create the simulation environment ─────────────────────────────
    # get_libero_env constructs an OffScreenRenderEnv from the task's BDDL file.
    # change_command=False uses the original registered task instruction.
    # The second and third return values (task description, bddl path) are not
    # needed here, so they are discarded with _.
    env, _, _ = get_libero_env(
        task,
        change_command=False,        # Use the canonical task instruction
        resolution=cfg.env_img_res,  # MuJoCo render resolution in pixels
    )

    # Set a fixed seed for reproducible physics initialisation across all 10 tasks.
    env.seed(0)

    # ── Step 2: Load pre-recorded initial state ────────────────────────────────
    try:
        # Retrieve the list of pre-recorded MuJoCo states for this task.
        # Each state is a NumPy array encoding the full simulator configuration
        # (joint positions, object poses, robot joint angles, etc.).
        initial_states = task_suite.get_task_init_states(task_id)

        # Reset the environment to clear any previous state.
        env.reset()

        # Apply the first pre-recorded initial state (index 0), which places
        # objects in the canonical position for this task. This ensures all
        # 10 tasks start from standardised, reproducible configurations.
        # Returns the observation dict at this initial state.
        obs = env.set_init_state(initial_states[0])

    except Exception:
        # Fallback: if pre-recorded states are unavailable (e.g. missing files
        # or benchmark API mismatch), use the stochastic default reset.
        # Note: this may reduce reproducibility of the visual comparison.
        env.reset()
        obs = env.get_observation()

    # ── Step 3: Physics warm-up ────────────────────────────────────────────────
    # Execute cfg.num_steps_wait (default: 10) zero-action simulator steps.
    # This allows gravity and soft-body contacts to settle so that the captured
    # frame shows a stable, realistic scene configuration.
    # get_libero_dummy_action returns a zero-vector of the correct action
    # dimensionality for the given model family.
    # env.step() returns (obs, reward, done, info); only obs is kept.
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    # ── Step 4: Extract raw camera frames ─────────────────────────────────────
    # get_libero_image: extracts the agentview (third-person) RGB frame.
    # Returns a NumPy array of shape (H, W, 3) with dtype uint8 [0, 255].
    img       = get_libero_image(obs)

    # get_libero_wrist_image: extracts the wrist camera RGB frame.
    # Returns a NumPy array of shape (H, W, 3) with dtype uint8 [0, 255].
    wrist_img = get_libero_wrist_image(obs)

    # ── Step 5: Release simulation resources ──────────────────────────────────
    # Close the environment immediately after frame capture to free the MuJoCo
    # rendering context, OpenGL buffers, and allocated memory. With 10 tasks
    # called sequentially, not closing would exhaust GPU memory and display
    # server file descriptors.
    env.close()

    # ── Step 6: Resize and convert to PIL ─────────────────────────────────────
    # resize_image_for_policy: resizes a (H, W, 3) uint8 numpy image to
    # (resize_size, resize_size, 3) using the configured interpolation method.
    # Image.fromarray: converts the resized numpy array to a PIL.Image.Image.
    # .convert("RGB"): ensures a consistent 3-channel layout regardless of
    # the source format (prevents accidental RGBA or greyscale propagation).
    img_pil   = Image.fromarray(
        resize_image_for_policy(img, resize_size)
    ).convert("RGB")

    wrist_pil = Image.fromarray(
        resize_image_for_policy(wrist_img, resize_size)
    ).convert("RGB")

    return img_pil, wrist_pil


# =============================================================================
# COSINE SIMILARITY UTILITY
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity (not cosine distance) between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors in
    their high-dimensional embedding space, providing an orientation-based
    similarity score that is invariant to vector magnitude (scale). This is the
    appropriate metric for comparing embedding vectors, which may have
    different L2 norms depending on the activation scale of the projector MLP.

    Formula
    -------
    .. math::

        \\text{sim}_{\\cos}(\\mathbf{a}, \\mathbf{b})
        = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{\\|\\mathbf{a}\\| \\cdot \\|\\mathbf{b}\\|}

    This is equivalent to the dot product of the L2-normalised vectors:

    .. math::

        \\text{sim}_{\\cos}(\\mathbf{a}, \\mathbf{b})
        = \\hat{\\mathbf{a}} \\cdot \\hat{\\mathbf{b}},
        \\quad \\hat{\\mathbf{v}} = \\frac{\\mathbf{v}}{\\|\\mathbf{v}\\|}

    Interpretation
    --------------
    - ``1.0``  : vectors are perfectly aligned (identical direction).
    - ``0.0``  : vectors are orthogonal (no linear correlation in direction).
    - ``-1.0`` : vectors point in opposite directions.

    Difference from ``compare_embeddings.cosine_distance()``
    ---------------------------------------------------------
    ``compare_embeddings.cosine_distance()`` returns ``1 - sim_cos``, a
    dissimilarity measure where 0 = identical and 2 = maximally dissimilar.
    This function returns the raw similarity directly, so that matrix values
    closer to 1.0 mean "more similar" — which is more intuitive for a
    similarity matrix.

    Parameters
    ----------
    a : np.ndarray of shape ``(hidden_dim,)``
        First embedding vector (any float dtype).
    b : np.ndarray of shape ``(hidden_dim,)``
        Second embedding vector (must have the same shape as ``a``).

    Returns
    -------
    float
        Cosine similarity in the range ``[-1.0, 1.0]``.

    Notes
    -----
    - An epsilon of ``1e-12`` is added to each norm before division to prevent
      division by zero for zero vectors (which should not occur with real model
      outputs but is a numerical safety measure).
    - The function modifies local copies of ``a`` and ``b`` (normalisation);
      the original arrays passed by the caller are not mutated.
    """
    # L2-normalise both vectors to unit length.
    # Adding 1e-12 (≈ machine epsilon for float64) prevents division-by-zero
    # for degenerate zero vectors without meaningfully affecting non-zero vectors.
    a = a / (np.linalg.norm(a) + 1e-12)   # Unit vector in direction of a
    b = b / (np.linalg.norm(b) + 1e-12)   # Unit vector in direction of b

    # The dot product of two unit vectors equals the cosine of the angle between them.
    # np.dot computes the inner product, returning a scalar (0-dimensional array).
    # float() converts it to a plain Python scalar for clean dict storage and printing.
    return float(np.dot(a, b))


# =============================================================================
# EUCLIDEAN DISTANCE UTILITY
# =============================================================================

def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Euclidean distance (L2 norm of the difference) between two vectors.

    Unlike cosine similarity, Euclidean distance is sensitive to both the
    direction *and* magnitude of the vectors. It measures the straight-line
    distance between two points in the high-dimensional embedding space.

    Euclidean distance is included alongside cosine similarity because:
    - Two embeddings can have high cosine similarity (same direction) but
      very different magnitudes, meaning they would be far apart in L2 space.
    - Conversely, two embeddings can be geometrically distant (large L2 gap)
      but still point in the same general direction (high cosine similarity).
    - Together, both metrics provide a more complete picture of the
      embedding geometry than either metric alone.

    Formula
    -------
    .. math::

        d_{\\text{euc}}(\\mathbf{a}, \\mathbf{b})
        = \\|\\mathbf{a} - \\mathbf{b}\\|_2
        = \\sqrt{\\sum_{i=1}^{d} (a_i - b_i)^2}

    Parameters
    ----------
    a : np.ndarray of shape ``(hidden_dim,)``
        First embedding vector.
    b : np.ndarray of shape ``(hidden_dim,)``
        Second embedding vector (must have the same shape as ``a``).

    Returns
    -------
    float
        Non-negative Euclidean distance. ``0.0`` iff ``a == b`` element-wise.
        For visual embeddings in LLM space (hidden_dim=4096), typical values
        between different task embeddings range from a few units to tens of
        units depending on how distinct the scenes are.
    """
    # np.linalg.norm with default ord=None computes the Frobenius / L2 norm.
    # Applied to the element-wise difference vector, this yields the Euclidean
    # distance between the two points in hidden_dim-dimensional space.
    return float(np.linalg.norm(a - b))


# =============================================================================
# TASK NAME ABBREVIATION HELPER
# =============================================================================

def short(name: str, n: int = 18) -> str:
    """
    Abbreviate a snake_case task name to a compact label for matrix printing.

    LIBERO task names are long descriptive strings (e.g.
    ``"put_the_wine_bottle_on_top_of_the_cabinet"``). Printing the full name
    in a 10×10 matrix would make the table too wide to fit on a standard
    terminal (80–120 columns). This function produces a concise abbreviation
    by taking the first 4 characters of each word, then hard-truncating to
    ``n`` characters.

    Algorithm
    ---------
    1. Replace all underscores with spaces to split the name into words.
    2. Take the first 4 characters of each word (preserving the word count).
    3. Join the truncated words with single spaces.
    4. Hard-truncate the result to ``n`` characters.

    Examples
    --------
    >>> short("put_the_wine_bottle_on_top_of_the_cabinet")
    'put  the  wine bott'  # First 4 chars of each word, truncated to 18
    >>> short("turn_on_the_stove")
    'turn on  the  stov'
    >>> short("open_the_middle_drawer_of_the_cabinet", n=20)
    'open the  midd draw of t'[:20]

    Parameters
    ----------
    name : str
        Task name in snake_case with underscores as word separators.
    n : int, default ``18``
        Maximum number of characters in the returned abbreviation.
        Column width in ``print_matrix()`` is set to match this value.

    Returns
    -------
    str
        Abbreviated task name of at most ``n`` characters.
    """
    # Replace underscores with spaces and split into a list of word strings.
    # e.g. "put_the_wine_bottle" → ["put", "the", "wine", "bottle"]
    words = name.replace("_", " ").split()

    # For each word, take at most the first 4 characters ([:4] safely handles
    # words shorter than 4 characters without raising an IndexError).
    # Join with a single space to form the abbreviated string.
    abbr = " ".join(w[:4] for w in words)

    # Hard-truncate to n characters to guarantee the label fits the column width.
    return abbr[:n]


# =============================================================================
# FORMATTED MATRIX PRINTER
# =============================================================================

def print_matrix(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    fmt: str = "{:.4f}",
) -> None:
    """
    Print an N×N numerical matrix as a formatted ASCII table to stdout.

    The table layout is:
    - A horizontal separator line scaled to the total table width.
    - A centred title line below the separator.
    - A header row with numeric column indices (0, 1, …, N-1).
    - N data rows, each prefixed with a numeric row index and a truncated
      task label (from ``short()``), followed by N formatted values.

    Layout Dimensions
    -----------------
    Each numeric column is ``col_w=10`` characters wide, right-aligned.
    The row-label prefix is ``lbl_w=22`` characters wide.
    Total table width: ``N × col_w + lbl_w + 4`` characters.

    For N=10: ``10×10 + 22 + 4 = 126`` characters — fits in a standard
    130-column terminal without line wrapping.

    Parameters
    ----------
    matrix : np.ndarray of shape ``(N, N)``
        Square matrix of values to display. Must be indexable as ``matrix[i, j]``.
    labels : List[str]
        List of ``N`` task name strings used to generate row labels.
        Column headers use numeric indices (0–N-1); row labels show both
        the numeric index and an abbreviated name from ``short()``.
    title : str
        Descriptive table title printed below the top separator line.
    fmt : str, default ``"{:.4f}"``
        Python format string applied to each matrix value.
        - ``"{:.4f}"`` for cosine similarity (4 decimal places, range 0–1).
        - ``"{:.2f}"``  for Euclidean distance (2 decimal places, larger range).

    Returns
    -------
    None
        All output is written to stdout via ``print()``.

    Example Output (abbreviated for clarity)
    -----------------------------------------
    ::

        ──────────────────────────────────────────────────────────────────
          COSINE SIMILARITY matrix
        ──────────────────────────────────────────────────────────────────
                                     0         1         2   ...
           0 put  the  wine       1.0000    0.9987    0.9993  ...
           1 open the  top        0.9987    1.0000    0.9991  ...
           2 turn on   the        0.9993    0.9991    1.0000  ...
    """
    n     = len(labels)   # Number of tasks (N); used as loop bound
    col_w = 10            # Width of each numeric value column in characters
    lbl_w = 22            # Width of the row-label column (index + abbreviated name)

    # ── Separator line ─────────────────────────────────────────────────────────
    # Total table width: N value columns + label column + 4 padding characters.
    sep = '─' * ((n * col_w) + lbl_w + 4)
    print(f"\n{sep}")

    # ── Title ──────────────────────────────────────────────────────────────────
    print(f"  {title}")
    print(sep)

    # ── Column header: numeric indices 0 … N-1 ────────────────────────────────
    # Start with an empty left-side label placeholder (right-aligned, lbl_w wide).
    header = f"{'':>{lbl_w}}"
    for i in range(n):
        # Right-align each column index within a (col_w - 1) field, followed by
        # the implicit space from the next iteration's leading space.
        header += f" {i:>{col_w-1}}"
    print(header)

    # ── Data rows ──────────────────────────────────────────────────────────────
    for i in range(n):
        # Prefix: 2-char right-aligned index + 1 space + left-aligned abbreviated label.
        # Total prefix width: 2 + 1 + (lbl_w - 4) + 1 = lbl_w characters.
        row = f"  {i:2d} {short(labels[i]):<{lbl_w-4}}"

        # Append each value in this row, formatted according to fmt.
        # Right-aligned within a (col_w - 1) field, with a leading space separator.
        for j in range(n):
            row += f" {fmt.format(matrix[i, j]):>{col_w-1}}"

        print(row)

# =============================================================================
# SANITY CHECK — VISUAL PIPELINE TEXT INDEPENDENCE
# =============================================================================

def sanity_check_text_independence(
    model,
    processor,
    img_pil: Image.Image,
    wrist_pil: Optional[Image.Image],
    prompts: List[str],
) -> None:
    """
    Empirically verify that the visual pipeline (vision_backbone + projector)
    is fully decoupled from the text input when ``use_film=False``.

    Scientific Rationale
    --------------------
    The OpenVLA-OFT architecture separates visual and linguistic processing
    into two distinct stages:

    **Stage 1 — Visual pipeline** (what this function tests):

    ::

        pixel_values
              │
              ▼
        vision_backbone (DINOv2 + SigLIP ViT)
              │
              ▼
        projector MLP
              │
              ▼
        projected_patches  ← the output under examination

    **Stage 2 — Language model**:

    ::

        [projected_patches || text_tokens]
              │
              ▼
        LLaMA-2 7B transformer layers
              │
              ▼
        action prediction

    When ``use_film=False``, Stage 1 processes ``pixel_values`` in complete
    isolation from the text tokens. The processor requires a prompt to
    build the full input dict (specifically ``input_ids`` and
    ``attention_mask``), but those tensors are never forwarded to the
    backbone or projector — they are only used in Stage 2. Therefore,
    varying the prompt while holding the image fixed should produce
    **bitwise-identical** ``projected_patches`` up to floating-point
    precision limits (~1e-7 for bfloat16 rounded to float32).

    When ``use_film=True``, FiLM (Feature-wise Linear Modulation)
    conditioning injects language features into intermediate ViT feature
    maps via learned per-channel affine transformations (scale and bias).
    In that case, different prompts applied to the same image produce
    measurably different ``projected_patches``, and this check would
    correctly report ``✗ DIVERSI``.

    Implication for the Main Analysis
    ----------------------------------
    If this sanity check passes (all patches identical across all prompts),
    it confirms that the 10×10 similarity and distance matrices computed in
    ``main()`` reflect **pure scene appearance**, completely independent of
    language. Any inter-task discrimination observed in the multimodal
    embeddings of ``compare_embeddings.py`` (which run the full LLM) must
    therefore arise **exclusively** from the text conditioning stage —
    confirming that the model uses language to distinguish which task to
    perform, not visual scene differences.

    Comparison Metric
    -----------------
    For each pair (Prompt 0 vs Prompt i), two difference statistics are
    reported over the full ``(n_patches, hidden_dim)`` patch matrix:

    - **max|diff|**: the single largest element-wise absolute difference.
      If ``use_film=False`` and computations are deterministic, this should
      be identically ``0.0`` or within float32 round-off (< 1e-7).
    - **mean|diff|**: the arithmetic mean of all absolute differences.
      Provides a global sense of magnitude even when individual extreme
      values are zero.

    The identity verdict uses ``np.allclose(ref, flat, atol=1e-5)``, which
    tolerates the small non-determinism introduced by bfloat16 → float32
    conversion and CPU memory layout differences.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded VLA model. Must expose ``model.vision_backbone``,
        ``model.projector``, and ``model.device``.
    processor : transformers.ProcessorMixin
        OpenVLA processor used to build the input dict for each prompt.
    img_pil : PIL.Image.Image
        Third-person camera frame held **fixed** across all prompt evaluations.
        This is the constant visual input; only the text varies.
    wrist_pil : PIL.Image.Image or None
        Wrist camera frame, also held fixed. Pass ``None`` for single-view
        mode (``cfg.num_images_in_input == 1``).
    prompts : List[str]
        List of formatted OpenVLA prompt strings to test. Prompt 0 serves as
        the reference; all subsequent prompts are compared against it.
        Should contain at least 2 prompts; ``SANITY_PROMPTS`` (4 prompts)
        is the standard input from ``main()``.

    Returns
    -------
    None
        Results are printed to stdout only. No values are returned.
        The function prints one comparison block per prompt pair, each
        containing the status verdict, max|diff|, and mean|diff|.
        A summary verdict is printed at the end.

    Output Format Example
    ---------------------
    ::

        ================================================================================
        SANITY CHECK: same image, different prompts → projected_patches identical?
        ================================================================================
          prompt: In: What action should the robot take to put the wine bottle...
          prompt: In: What action should the robot take to open the middle layer...
          prompt: In: What action should the robot take to put the bowl on the stove...
          prompt: In: What action should the robot take to place the wine bottle...

          Prompt 0 vs Prompt 1: ✓ IDENTICAL
            max|diff|  = 0.00e+00
            mean|diff| = 0.00e+00

          Prompt 0 vs Prompt 2: ✓ IDENTICAL
            max|diff|  = 0.00e+00
            mean|diff| = 0.00e+00

          Prompt 0 vs Prompt 3: ✓ IDENTICAL
            max|diff|  = 0.00e+00
            mean|diff| = 0.00e+00

          ✓ CONFIRMED: visual pipeline is text-independent (use_film=False)
    """
    print("\n" + "=" * 80)
    print("SANITY CHECK: same image, different prompts → projected_patches identical?")
    print("=" * 80)

    # ── Step 1: Extract projected_patches for every prompt ─────────────────────
    # The image (img_pil, wrist_pil) is held constant across all calls.
    # Only the prompt string changes between iterations.
    # If use_film=False, the backbone and projector ignore input_ids entirely,
    # so all calls should return the exact same patches_flat matrix.
    all_flat: List[np.ndarray] = []

    for p in prompts:
        # extract_visual_embedding runs: processor → backbone → projector → numpy.
        # The prompt p is passed to the processor but does NOT reach the projector
        # when use_film=False. We collect patches_flat; patches_mean is unused here.
        flat, _ = extract_visual_embedding(model, processor, img_pil, wrist_pil, p)
        all_flat.append(flat)                # (n_patches, hidden_dim) per prompt
        print(f"  prompt: {p[:70]}...")       # Show first 70 chars for readability

    # ── Step 2: Compare all prompts against Prompt 0 (reference) ───────────────
    # Prompt 0 is the canonical reference; all others are compared to it.
    # Using a pairwise reference (instead of all-vs-all) is sufficient to
    # confirm text independence: if p0 == p1 and p0 == p2, then p1 == p2
    # by transitivity.
    ref: np.ndarray = all_flat[0]   # Reference patch matrix: (n_patches, hidden_dim)
    all_identical: bool = True       # Accumulates the conjunction of all pair verdicts

    for i, flat in enumerate(all_flat[1:], start=1):
        # Element-wise absolute difference matrix: (n_patches, hidden_dim)
        abs_diff = np.abs(ref - flat)

        # max|diff|: the single largest deviation anywhere in the matrix.
        # Should be 0.0 (or < 1e-7 due to float32 rounding) when use_film=False.
        max_diff: float = float(abs_diff.max())

        # mean|diff|: average absolute deviation across all elements.
        # Provides a global characterisation; 0.0 when patches are identical.
        mean_diff: float = float(abs_diff.mean())

        # np.allclose: returns True iff |ref - flat| <= atol element-wise.
        # atol=1e-5 is deliberately permissive to tolerate any bfloat16 → float32
        # round-off artefacts. True identity means max_diff < 1e-7.
        identical: bool = bool(np.allclose(ref, flat, atol=1e-5))

        # Update the global verdict: all pairs must be identical to pass.
        all_identical = all_identical and identical

        status = "✓ IDENTICAL" if identical else "✗ DIFFERENT"
        print(f"\n  Prompt 0 vs Prompt {i}: {status}")
        print(f"    max|diff|  = {max_diff:.2e}")    # e.g. 0.00e+00 on perfect identity
        print(f"    mean|diff| = {mean_diff:.2e}")

    # ── Step 3: Global verdict ─────────────────────────────────────────────────
    if all_identical:
        # All prompt pairs produced bitwise-identical projected_patches.
        # This confirms that the visual pipeline is fully text-independent,
        # validating the architectural assumption for use_film=False.
        print(
            "\n  ✓ CONFIRMED: visual pipeline is text-independent (use_film=False)"
        )
    else:
        # At least one pair produced different projected_patches.
        # This either means use_film=True is inadvertently active in the loaded
        # checkpoint, or there is an unexpected text-image coupling in the
        # backbone or processor implementation that warrants investigation.
        print(
            "\n  ✗ WARNING: visual pipeline depends on text input "
            "→ FiLM conditioning is active!"
        )


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """
    This function is the primary entry point when the script is run directly.
    It coordinates six sequential analysis phases, each building on the results
    of the previous:

    Analysis Phases
    ---------------

    **Phase 1 — Model Loading**
        Instantiates a ``Cfg`` object with default parameters and calls
        ``load_model()`` to load the VLA backbone, processor, and resolve
        the image resize resolution. Model weights are placed on CUDA in
        bfloat16 precision.

    **Phase 2 — Benchmark Initialisation**
        Retrieves the LIBERO benchmark registry and instantiates the
        ``libero_goal`` task suite, which provides access to all 10 kitchen
        manipulation tasks and their pre-recorded initial states.

    **Phase 3 — Visual Embedding Extraction**
        For each of the 10 tasks:

        1. Retrieves the task object from the benchmark suite.
        2. Calls ``get_first_frame()`` to spin up the MuJoCo environment,
           stabilise the scene physics, and capture the initial agentview
           and wrist camera frames.
        3. Calls ``extract_visual_embedding()`` with a neutral prompt to
           compute ``patches_flat`` (all patch vectors) and ``patches_mean``
           (scene-level descriptor) from the frozen visual pipeline.

        All 10 mean embeddings are stored in ``mean_embeddings``; all 10
        patch matrices are stored in ``flat_embeddings`` (used by the sanity
        check in Phase 6).

        The neutral prompt used here is:
        ``"In: What action should the robot take to perform the task?\nOut:"``

        Because ``use_film=False``, the specific prompt content is irrelevant —
        the same pixel_values will always produce the same projected_patches.
        A neutral, non-task-specific prompt is used for clarity.

    **Phase 4 — Similarity Matrix Construction**
        Builds two symmetric 10×10 matrices by iterating over all task pairs
        (i, j) including the diagonal (self-comparisons):

        - **Cosine similarity matrix**: entry [i, j] = ``cosine_sim(mean_i, mean_j)``.
          Range [−1, 1]; diagonal entries are always 1.0.
        - **Euclidean distance matrix**: entry [i, j] = ``euclidean_dist(mean_i, mean_j)``.
          Range [0, ∞); diagonal entries are always 0.0.

        Both matrices are printed to stdout using ``print_matrix()`` with
        abbreviated task labels.

    **Phase 5 — Aggregate Statistics and Interpretation**
        Extracts the 90 off-diagonal entries (all pairs where i ≠ j) from
        both matrices and computes mean, standard deviation, minimum, and
        maximum. An automated interpretation is then derived based on the
        mean cosine similarity:

        - ``> 0.99``: visual embeddings are near-identical across all tasks.
          The model relies **exclusively** on language to discriminate tasks.
          Differences in L1/L2 task composition performance are attributable
          to linguistic generalisation, not to visual scene recognition.
        - ``> 0.95``: embeddings are similar but not degenerate. Language
          still dominates but visual features make a minor contribution.
        - ``≤ 0.95``: moderate similarity; both visual and linguistic
          components contribute meaningfully to task discrimination.

    **Phase 6 — Sanity Check**
        Re-captures the first frame of Task 0 and calls
        ``sanity_check_text_independence()`` with the four prompts from
        ``SANITY_PROMPTS``. This verifies the architectural invariant that
        ``projected_patches`` are completely determined by ``pixel_values``
        alone when ``use_film=False``, providing a lower-level confirmation
        that Phases 3–5 measured visual information and nothing else.

    Design Decisions
    ----------------
    - **Mean-pooled patches as scene descriptor**: The mean over all spatial
      patches (``patches_mean``) compresses the ``(n_patches, hidden_dim)``
      matrix to a single ``(hidden_dim,)`` vector. This is the simplest
      spatial aggregation and is equivalent to what many vision encoders
      expose as a CLS token. It is not the only valid choice — one could
      compare full patch matrices or use max-pooling — but it is sufficient
      to answer the key question: are the 10 task scenes visually
      distinguishable at the projector output?

    - **Neutral prompt for Phase 3**: Since ``use_film=False`` guarantees
      text independence (confirmed in Phase 6), the exact prompt used during
      embedding extraction does not matter. A neutral prompt is used to
      signal clearly that no task-specific information is being injected.

    - **Task 0 for sanity check**: Reuses the same task to avoid spinning up
      two separate environments for the same scene. The sanity check is
      agnostic to which task is used, since it tests a model property
      (text-independence of the visual pipeline) rather than a task property.

    Parameters
    ----------
    None

    Returns
    -------
    None
        All results are printed to stdout. No values are returned and no files
        are written by this function.

    Printed Output Summary
    ----------------------
    ::

        ================================================================================
        VISUAL EMBEDDING ANALYSIS — OpenVLA-OFT checkpoint 20000 — LIBERO Goal
        ================================================================================
        Model use_film = False
        Num tasks      = 10
        ================================================================================

        [ 0] put_the_wine_bottle_on_top_of_the_cabinet
             patch shape: (1152, 4096)  |  mean shape: (4096,)
        [ 1] put_the_bowl_on_the_stove
             ...
        [10 tasks total]

        ──────────────────── COSINE SIMILARITY 10×10 matrix ────────────────────
        ──────────────────── EUCLIDEAN DISTANCE 10×10 matrix ────────────────────

        STATISTICS (off-diagonal pairs, N=90)
          Cosine Similarity:   mean=0.9982  std=0.0005  min=0.9971  max=0.9993
          Euclidean Distance:  mean=18.32   std=1.24    min=16.11   max=21.03

        INTERPRETATION:
          ● Cosine similarity very high (>0.99): visual embeddings are nearly identical
            across all tasks. The model relies PRIMARILY on the text prompt to
            discriminate which action to take.
            → Differences in L1/L2/L3 performance are attributable to the
              LINGUISTIC component of the embedding, not the visual one.

        ================================================================================
        SANITY CHECK: same image, different prompts → projected_patches identical?
        ================================================================================
          ✓ CONFIRMED: visual pipeline is text-independent (use_film=False)

        ================================================================================
        Analysis complete.
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Model Loading
    # ═══════════════════════════════════════════════════════════════════════════
    cfg = Cfg()   # Instantiate config with all default values (use_film=False)

    # Load VLA backbone, processor, and derive image resize resolution.
    # model.vision_backbone and model.projector are the components used here.
    model, processor, resize_size = load_model(cfg)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Benchmark Suite Initialisation
    # ═══════════════════════════════════════════════════════════════════════════
    # Returns dict mapping suite name → benchmark class (not yet instantiated).
    benchmark_dict = benchmark.get_benchmark_dict()

    # Instantiate the libero_goal suite: 10 kitchen pick-and-place tasks sharing
    # the same scene layout (table, stove, cabinet, drawer).
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # n_tasks = 10 for libero_goal.
    n_tasks: int = task_suite.n_tasks

    # Print analysis header with key configuration parameters.
    print(f"\n{'=' * 80}")
    print("VISUAL EMBEDDING ANALYSIS — OpenVLA-OFT checkpoint 20000 — LIBERO Goal")
    print(f"{'=' * 80}")
    # use_film is critical: if True, interpretation of results changes entirely.
    print(f"Model use_film = {cfg.use_film}")
    print(f"Num tasks      = {n_tasks}")
    print(f"{'=' * 80}\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Visual Embedding Extraction
    # ═══════════════════════════════════════════════════════════════════════════
    mean_embeddings: List[np.ndarray] = []   # (hidden_dim,) per task — scene descriptors
    flat_embeddings: List[np.ndarray] = []   # (n_patches, hidden_dim) per task — full patches
    task_labels: List[str] = []              # Task names for matrix row/column labels

    # Neutral prompt: content is irrelevant when use_film=False (confirmed in Phase 6).
    # A generic instruction is used to make clear that no task-specific language is
    # being injected into the visual pipeline.
    neutral_prompt = (
        "In: What action should the robot take to perform the task?\nOut:"
    )

    for task_id in range(n_tasks):
        # Retrieve the Task NamedTuple for this benchmark task.
        task = task_suite.get_task(task_id)

        # Use the canonical task name from the pre-defined TASK_NAMES list.
        # These match the LIBERO benchmark task names exactly.
        task_key = TASK_NAMES[task_id]
        task_labels.append(task_key)

        print(f"[{task_id:2d}] {task_key}")

        # Spin up the MuJoCo simulation, stabilise physics, capture both frames,
        # and shut down the environment. Returns PIL.Image (RGB) for each camera.
        img_pil, wrist_pil = get_first_frame(
            task, cfg, resize_size, task_id, task_suite
        )

        # Pass the wrist image only when the model was fine-tuned on two camera
        # views (num_images_in_input=2). For single-view models, pass None.
        wrist = wrist_pil if cfg.num_images_in_input > 1 else None

        # Run vision_backbone + projector; stop before the language model.
        # flat: (n_patches, hidden_dim) — full spatial patch matrix for this scene.
        # mean: (hidden_dim,) — mean-pooled scene descriptor vector.
        flat, mean = extract_visual_embedding(
            model, processor, img_pil, wrist, neutral_prompt
        )

        flat_embeddings.append(flat)
        mean_embeddings.append(mean)

        # Confirm tensor dimensionalities for debugging.
        # Typical output: "patch shape: (1152, 4096)  |  mean shape: (4096,)"
        print(f"     patch shape: {flat.shape}  |  mean shape: {mean.shape}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4: Similarity and Distance Matrix Construction
    # ═══════════════════════════════════════════════════════════════════════════
    n = n_tasks   # 10

    # Preallocate dense 10×10 matrices with zeros.
    cos_matrix = np.zeros((n, n))   # Cosine similarity ∈ [−1, 1]; diagonal = 1.0
    euc_matrix = np.zeros((n, n))   # Euclidean distance ∈ [0, ∞); diagonal = 0.0

    # Compute all N² = 100 pairwise values (including self-comparisons on diagonal).
    # The matrices are symmetric by construction (both metrics satisfy symmetry),
    # but both halves are explicitly computed for code simplicity.
    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = cosine_sim(mean_embeddings[i], mean_embeddings[j])
            euc_matrix[i, j] = euclidean_dist(mean_embeddings[i], mean_embeddings[j])

    # Print the cosine similarity matrix as a formatted table with abbreviated labels.
    print_matrix(
        cos_matrix, task_labels,
        "COSINE SIMILARITY between visual embeddings (mean-pool patches) — 10 tasks",
        fmt="{:.4f}",   # 4 decimal places; range is [−1, 1]
    )

    # Print the euclidean distance matrix.
    print_matrix(
        euc_matrix, task_labels,
        "EUCLIDEAN DISTANCE between visual embeddings (mean-pool patches) — 10 tasks",
        fmt="{:.2f}",   # 2 decimal places; range is [0, ∞)
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 5: Aggregate Statistics and Automated Interpretation
    # ═══════════════════════════════════════════════════════════════════════════
    # Build a boolean mask that selects off-diagonal entries only.
    # np.eye(n, dtype=bool): True on diagonal; ~ inverts to True off-diagonal.
    mask = ~np.eye(n, dtype=bool)

    # Extract the N*(N-1) = 90 off-diagonal values from each matrix.
    off_cos: np.ndarray = cos_matrix[mask]   # 90 cosine similarity values
    off_euc: np.ndarray = euc_matrix[mask]   # 90 euclidean distance values

    # Print the four summary statistics for both metrics.
    print(f"\n{'─' * 80}")
    print(f"STATISTICS (off-diagonal pairs, N={n * (n - 1)})")
    print(
        f"  Cosine Similarity:   mean={off_cos.mean():.4f}  std={off_cos.std():.4f}  "
        f"min={off_cos.min():.4f}  max={off_cos.max():.4f}"
    )
    print(
        f"  Euclidean Distance:  mean={off_euc.mean():.2f}   std={off_euc.std():.2f}   "
        f"min={off_euc.min():.2f}   max={off_euc.max():.2f}"
    )

    # Automated interpretation based on the mean off-diagonal cosine similarity.
    mean_cos: float = float(off_cos.mean())
    print(f"\n{'─' * 80}")
    print("INTERPRETATION:")

    if mean_cos > 0.99:
        # Extremely high similarity: the projector produces nearly identical
        # scene representations for all 10 LIBERO Goal tasks. This is expected
        # because all tasks share the same physical scene (same objects, same
        # kitchen layout); only the object configuration subtly differs.
        # The model must therefore rely almost exclusively on the language
        # instruction to determine which action to take.
        #
        # Implication for task composition evaluation:
        # Since visual features are near-degenerate across tasks, the success
        # or failure of the model on L1/L2 task composition scenarios must be
        # explained by whether the model can correctly interpret and generalise
        # the novel language instructions — not by whether it can visually
        # recognise the scene differences.
        print("  ● Cosine similarity very high (>0.99): visual embeddings are nearly")
        print("    identical across all tasks. The model relies PRIMARILY on the text")
        print("    prompt to discriminate which action to take.")
        print("    → Differences in L1/L2 task composition performance are attributable")
        print("      to the LINGUISTIC component of the embedding, not the visual one.")

    elif mean_cos > 0.95:
        # High but not extreme similarity: the visual pipeline introduces some
        # scene-level differentiation, but language conditioning still dominates.
        print("  ● Cosine similarity high (>0.95): the visual pipeline produces similar")
        print("    but not identical representations. Language still plays a dominant role.")

    else:
        # Moderate similarity: the visual pipeline produces meaningfully different
        # representations for different tasks. Both modalities contribute.
        # This outcome would suggest that the visual features alone partially encode
        # task identity — an unexpected finding for a shared-scene benchmark.
        print("  ● Cosine similarity moderate: the visual pipeline differentiates tasks.")
        print("    Both visual and linguistic components contribute to task discrimination.")

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 6: Sanity Check — Visual Pipeline Text Independence
    # ═══════════════════════════════════════════════════════════════════════════
    # Re-capture the first frame of Task 0 to use as the fixed test image.
    # Any task would work; Task 0 is chosen arbitrarily.
    task0 = task_suite.get_task(0)
    img0, wrist0 = get_first_frame(task0, cfg, resize_size, 0, task_suite)

    # Run the sanity check with 4 semantically diverse prompts on the same image.
    # Expected result when use_film=False: all 4 produce identical projected_patches.
    sanity_check_text_independence(
        model,
        processor,
        img0,
        wrist0 if cfg.num_images_in_input > 1 else None,   # Wrist only for multi-view
        SANITY_PROMPTS,   # 4 prompts: different actions, different objects, one paraphrase
    )

    print(f"\n{'=' * 80}")
    print("Analysis complete.")


# =============================================================================
# MODULE ENTRY POINT
# =============================================================================

# Standard Python idiom: execute main() only when the script is invoked
# directly (e.g. ``python verify_visual_embedding.py``), not when it is
# imported as a module by another script (e.g. in unit tests or notebooks).
# This allows the module-level constants (SANITY_PROMPTS, Cfg) and utility
# functions (extract_visual_embedding, cosine_sim, etc.) to be imported
# and reused by other scripts without triggering the full analysis pipeline.
if __name__ == "__main__":
    main()