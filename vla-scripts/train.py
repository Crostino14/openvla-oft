"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of
pretrained Vision-Language Models (VLMs). The model is trained using mixtures
of the Open-X Embodiment (OXE) dataset — a large-scale, diverse collection of
robotic manipulation demonstrations.

Training is performed in native PyTorch, leveraging Fully-Sharded Data Parallel
(FSDP) to distribute computation across multiple GPUs and/or nodes. FSDP shards
both model parameters and optimizer states across devices, significantly reducing
per-device memory usage compared to standard Data Parallel (DP) or vanilla DDP.

By default, this script assumes CUDA toolkit >= 11.0 to enable BF16 (Brain Float
16) mixed-precision training, which provides a wider dynamic range than FP16 while
maintaining the memory savings of reduced precision.

Architecture Overview:
    - Vision Backbone: Encodes raw image observations into visual feature embeddings
      (e.g., DINOv2 + SigLIP dual encoder, processed at 224px resolution).
    - LLM Backbone: A large autoregressive language model that processes both visual
      embeddings (via a learned projector) and tokenized language instructions.
    - Action Head: The LLM's output tokens are decoded into discrete robot actions
      using a learned `action_tokenizer`.

Notes & Prerequisites:
    - To set a custom cache location for HuggingFace / TIMM model artifacts:
        export HF_HOME="<PATH>"   # add to ~/.bashrc for persistence
        Example: export HF_HOME="/mnt/fsx/skaramcheti/cache"
    - To suppress verbose TensorFlow logs (common when TF is installed alongside PT):
        export TF_CPP_MIN_LOG_LEVEL=3

Run Commands:
    Single Node, 1 GPU  (debug mode):
        torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py

    Single Node, K GPUs:
        torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py

Dependencies:
    - torch >= 2.0 (for FSDP v2 support and BF16)
    - draccus: Structured configuration management via dataclasses
    - prismatic: Internal library providing VLA models, datasets, and training utilities
    - yaml, json: Standard serialization for configuration persistence
    - torch.distributed: PyTorch's native distributed communication backend

References:
    - OpenVLA Paper: https://arxiv.org/abs/2406.09246
    - Open-X Embodiment Dataset: https://robotics-transformer-x.github.io/
    - PyTorch FSDP Docs: https://pytorch.org/docs/stable/fsdp.html
"""

# ─────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────
import json          # For serializing the training configuration to JSON format
import os            # For filesystem operations: makedirs, environ access
import re            # For regex-based extraction of step/epoch from checkpoint filenames
from dataclasses import dataclass, field   # dataclass: declarative config; field: default_factory support
from pathlib import Path                   # Object-oriented filesystem path handling
from typing import Optional, Tuple, Union  # Type hints for optional, tuple, and union types

# ─────────────────────────────────────────────
# Third-Party Library Imports
# ─────────────────────────────────────────────
import draccus                        # Structured CLI + YAML config system built on dataclasses
import torch                          # Core PyTorch tensor library and neural network framework
import torch.distributed as dist      # PyTorch distributed communication primitives (barriers, process groups)
import yaml                           # YAML parser for loading/dumping config files

# ─────────────────────────────────────────────
# Internal `prismatic` Library Imports
# ─────────────────────────────────────────────
from prismatic.conf import VLAConfig, VLARegistry
# VLAConfig: Dataclass holding all hyperparameters and architecture choices for a VLA model.
# VLARegistry: Enum-like registry mapping named VLA configurations (e.g., DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS)
#              to their corresponding VLAConfig instances.

from prismatic.models import load, load_vla
# load: Loads a pretrained VLM (Vision-Language Model) from a HuggingFace model ID or local path.
# load_vla: Loads a previously trained/finetuned VLA checkpoint for continued training or inference.

from prismatic.overwatch import initialize_overwatch
# initialize_overwatch: Creates a distributed-aware logger wrapper around Python's logging.Logger.
#                       On rank 0, it logs to stdout/file; on other ranks, logging is suppressed.

from prismatic.training import VLAMetrics, get_train_strategy
# VLAMetrics: Handles real-time metric tracking and logging to configured backends (JSONL, W&B).
# get_train_strategy: Factory function that returns the appropriate training loop object
#                     (e.g., FSDP-based strategy) based on the config.

from prismatic.util import set_global_seed
# set_global_seed: Sets the random seed for Python's random, NumPy, and PyTorch (CPU + CUDA)
#                  to ensure reproducibility. Can optionally return a worker_init_fn for DataLoaders.

from prismatic.vla import get_vla_dataset_and_collator
# get_vla_dataset_and_collator: Constructs the RLDS-based Open-X Embodiment dataset pipeline,
#                               the action tokenizer, and the batch collation function.

from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
# save_dataset_statistics: Persists per-dataset action normalization statistics (mean, std, min, max)
#                          to disk so they can be reloaded at inference time for de-normalization.


# ─────────────────────────────────────────────
# Global Environment Configuration
# ─────────────────────────────────────────────

# Disable HuggingFace tokenizer parallelism to prevent deadlocks when the DataLoader
# spawns multiple worker processes. Tokenizers use Rust-based thread pools internally;
# forking after their initialization can cause hangs.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ─────────────────────────────────────────────
# Logger Initialization
# ─────────────────────────────────────────────

# Initialize the global distributed-aware logger (Overwatch) for this module.
# `__name__` resolves to "train" (or the package path if imported), which becomes
# the logger's name. This object provides `.info()`, `.warning()`, `.error()` methods
# as well as distributed helpers like `.is_rank_zero()` and `.world_size()`.
overwatch = initialize_overwatch(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TrainConfig: Top-Level Training Configuration Dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """
    Top-level configuration dataclass for VLA training runs.

    This dataclass aggregates all training hyperparameters, filesystem paths,
    checkpoint settings, logging preferences, and distributed training parameters
    into a single structured object. It is parsed from the command line and/or a
    YAML configuration file via the `draccus` library.

    The `__post_init__` method automatically promotes key optimization parameters
    from the nested `VLAConfig` (`self.vla`) to the top level for convenience, and
    validates that the number of launched GPU processes matches the expected world size
    declared in the VLA configuration.

    Attributes:
        vla (VLAConfig):
            Nested configuration object holding all VLA-specific hyperparameters
            (architecture choices, dataset mixture, optimizer settings, etc.).
            Default: DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS configuration.

        data_root_dir (Path):
            Filesystem path to the root directory containing the Open-X Embodiment
            (OXE) dataset, organized in RLDS (Reinforcement Learning Dataset Spec) format.
            Default: "datasets/open-x-embodiment"

        run_root_dir (Path):
            Filesystem path to the root directory where training artifacts are stored.
            Each run creates a subdirectory `{run_root_dir}/{run_id}/` containing
            checkpoints, metrics logs, and configuration files.
            Default: "runs"

        pretrained_checkpoint (Optional[Path]):
            Absolute path to a `.pt` VLA checkpoint file for resuming training.
            If None, training starts from a base pretrained VLM (specified by `vla.base_vlm`).
            Default: None

        is_resume (bool):
            If True and `pretrained_checkpoint` is set, the script validates that the
            checkpoint filename encodes the expected `resume_step` and `resume_epoch`.
            This acts as an explicit sanity check to prevent accidentally loading
            the wrong checkpoint.
            Default: True

        resume_step (Optional[int]):
            The global optimizer step at which training was previously interrupted.
            Must match the step encoded in the checkpoint filename when `is_resume=True`.
            Default: None

        resume_epoch (Optional[int]):
            The epoch at which training was previously interrupted.
            Must match the epoch encoded in the checkpoint filename when `is_resume=True`.
            Default: None

        run_id (Optional[str]):
            Unique string identifier for the current training run, used as the directory
            name under `run_root_dir` and as the W&B run name.
            Auto-generated as `{vla_id}+n{nodes}+b{batch_size}+x{seed}` if not specified.
            Default: None

        run_id_note (Optional[str]):
            Optional free-form string appended to `run_id` after a `--` separator.
            Useful for adding human-readable annotations to a run (e.g., "ablation-lr1e4").
            Default: None

        save_interval (int):
            Number of optimizer steps between checkpoint saves.
            Default: 2500

        image_aug (bool):
            Whether to apply random image augmentations (e.g., color jitter, random crops)
            to training images. Helps improve robustness to visual domain shifts.
            Default: False

        seed (int):
            Master random seed for reproducibility. Applied to Python's `random`,
            NumPy, and both CPU/CUDA PyTorch RNGs.
            Default: 7

        hf_token (Union[str, Path]):
            HuggingFace API token for accessing gated model repositories (e.g., LLaMA).
            If a `Path` is provided, the token is read from that file.
            If a `str` is provided, it is treated as an environment variable name.
            Default: Path(".hf_token")

        trackers (Tuple[str, ...]):
            Tuple of tracker backend names to initialize for metric logging.
            Supported values: "jsonl" (local file), "wandb" (Weights & Biases).
            Default: ("jsonl", "wandb")

        wandb_project (str):
            Name of the Weights & Biases project under which this run is logged.
            Default: "openvla"

        wandb_entity (str):
            Weights & Biases entity (user or team) under which the project lives.
            Default: "stanford-voltron"

        # --- Promoted optimization parameters (set in __post_init__) ---
        epochs (int):               Total number of training epochs (from vla.epochs).
        max_steps (int):            Maximum number of optimizer steps (from vla.max_steps).
        global_batch_size (int):    Total effective batch size across all GPUs (from vla.global_batch_size).
        per_device_batch_size (int):Number of samples per GPU per forward pass (from vla.per_device_batch_size).
        learning_rate (float):      Peak learning rate for the optimizer (from vla.learning_rate).
        weight_decay (float):       L2 regularization coefficient (from vla.weight_decay).
        max_grad_norm (float):      Maximum gradient norm for gradient clipping (from vla.max_grad_norm).
        lr_scheduler_type (str):    Learning rate schedule type, e.g. "linear", "cosine" (from vla).
        warmup_ratio (float):       Fraction of total steps used for linear LR warm-up (from vla).
        train_strategy (str):       Name of the FSDP training strategy to use (from vla.train_strategy).
    """

    # fmt: off  # Disable Black auto-formatting for the field block to preserve alignment

    # ── VLA Model Configuration ──────────────────────────────────────────────
    # Nested VLAConfig object. Uses default_factory to call VLAConfig.get_choice_class()
    # which returns a class constructor for the specified VLA configuration preset.
    # The default preset is DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS, which uses a
    # dual vision encoder (DINOv2 + SigLIP) at 224px, trained on the full OXE mixture.
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # ── Filesystem Paths ──────────────────────────────────────────────────────
    data_root_dir: Path = Path(                  # Root path to OXE dataset (RLDS format)
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")            # Root path for saving runs (logs + checkpoints)

    # ── Checkpoint / Resume Parameters ────────────────────────────────────────
    pretrained_checkpoint: Optional[Path] = None # Path to .pt checkpoint file; None = start fresh
    is_resume: bool = True                       # True = validate that checkpoint step/epoch match resume_*
    resume_step: Optional[int] = None            # Expected global step in checkpoint filename
    resume_epoch: Optional[int] = None           # Expected epoch in checkpoint filename

    # ── Run Identity & Behavior ───────────────────────────────────────────────
    run_id: Optional[str] = None                 # Run identifier (auto-generated if None)
    run_id_note: Optional[str] = None            # Optional suffix appended to run_id
    save_interval: int = 2500                    # Steps between checkpoint saves
    image_aug: bool = False                      # Enable random image augmentations
    seed: int = 7                                # Global random seed

    # ── HuggingFace Authentication ────────────────────────────────────────────
    hf_token: Union[str, Path] = Path(".hf_token")  # Path to file or env var name holding HF token

    # ── Experiment Tracking ───────────────────────────────────────────────────
    trackers: Tuple[str, ...] = ("jsonl", "wandb")  # Active logging backends
    wandb_project: str = "openvla"                  # W&B project name
    wandb_entity: str = "stanford-voltron"          # W&B team/user entity

    # fmt: on

    def __post_init__(self) -> None:
        """
        Post-initialization hook that promotes VLA optimization parameters to the
        top-level config and validates distributed training world size.

        This method is called automatically by Python's dataclass machinery
        immediately after `__init__`. It performs two tasks:

        1. **Parameter Promotion**: Copies key training hyperparameters from the
           nested `self.vla` (VLAConfig) object to `self` directly. This avoids
           having to write `cfg.vla.learning_rate` everywhere in the training loop;
           instead, `cfg.learning_rate` is available directly.

           Promoted parameters:
               - epochs           : Total number of full passes over the dataset.
               - max_steps        : Hard cap on optimizer steps (overrides epochs if set).
               - global_batch_size: Effective batch size = per_device_batch_size × world_size × grad_accum.
               - per_device_batch_size: Samples per GPU per forward pass.
               - learning_rate    : Peak LR for AdamW (or chosen optimizer).
               - weight_decay     : L2 penalty coefficient.
               - max_grad_norm    : Gradient clipping threshold (prevents exploding gradients).
               - lr_scheduler_type: Type of LR decay schedule (e.g., "cosine", "linear").
               - warmup_ratio     : Fraction of total steps for linear warm-up phase.
               - train_strategy   : Identifier for the FSDP training strategy class.

        2. **World Size Validation**: Asserts that the number of GPU processes launched
           by `torchrun` matches `self.vla.expected_world_size`. This prevents
           misconfigured multi-node launches (e.g., forgetting to set `--nproc-per-node`).

        Raises:
            AssertionError: If `overwatch.world_size()` does not equal
                            `self.vla.expected_world_size`.

        Returns:
            None
        """
        # ── Promote optimizer hyperparameters from vla sub-config ──────────
        self.epochs = self.vla.epochs                           # Total training epochs
        self.max_steps = self.vla.max_steps                     # Optional hard step limit
        self.global_batch_size = self.vla.global_batch_size     # Total across all GPUs
        self.per_device_batch_size = self.vla.per_device_batch_size  # Per-GPU batch size

        self.learning_rate = self.vla.learning_rate             # Peak learning rate
        self.weight_decay = self.vla.weight_decay               # AdamW weight decay
        self.max_grad_norm = self.vla.max_grad_norm             # Gradient clipping norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type     # LR schedule type
        self.warmup_ratio = self.vla.warmup_ratio               # Warm-up phase fraction

        self.train_strategy = self.vla.train_strategy           # FSDP strategy name

        # ── Validate distributed world size ────────────────────────────────
        # `overwatch.world_size()` queries `dist.get_world_size()` — the number of
        # processes in the current process group (i.e., total GPUs across all nodes).
        # This must equal the value declared in VLAConfig to avoid training with
        # incorrect effective batch sizes or gradient accumulation factors.
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"


# ══════════════════════════════════════════════════════════════════════════════
# train: Main Training Entry Point
# ══════════════════════════════════════════════════════════════════════════════

@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    """
    Main training function for VLA (Vision-Language-Action) policies.

    Orchestrates the full training pipeline, from environment setup and model loading,
    through dataset construction and training strategy initialization, to the main
    training loop and final cleanup. This function is decorated with `@draccus.wrap()`,
    which automatically parses `TrainConfig` from command-line arguments and/or a YAML
    config file before calling this function.

    The function assumes it is launched via `torchrun`, which spawns one process per GPU.
    Each process executes this function independently but they coordinate through
    `torch.distributed` (initialized automatically by `overwatch` / `torchrun`).

    High-Level Pipeline:
        1. Initialize CUDA device for this process and clear the GPU memory cache.
        2. Build a unique `run_id` string and create run/checkpoint directories.
        3. Serialize the resolved configuration to both YAML and JSON (rank 0 only).
        4. Load the VLA model (either from a base VLM or from a prior checkpoint).
        5. Validate that all model parameters are in FP32 (required before FSDP wrapping).
        6. Determine the training stage (which components are frozen vs. trainable).
        7. Freeze the appropriate model backbones and log the frozen/unfrozen parameters.
        8. Construct the Open-X Embodiment dataset, action tokenizer, and data collator.
        9. Save dataset normalization statistics to disk (rank 0 only).
        10. Initialize the FSDP training strategy with all optimizer/scheduler settings.
        11. Initialize metrics tracking (JSONL + optional Weights & Biases).
        12. Execute the main VLA training loop.
        13. Finalize metrics, synchronize all processes, and tear down the process group.

    Args:
        cfg (TrainConfig):
            Fully resolved training configuration object. All fields are populated
            either from their defaults, a YAML config file, or CLI overrides.
            Automatically injected by `@draccus.wrap()`.

    Returns:
        None

    Raises:
        AssertionError:
            - If `cfg.is_resume=True` and the checkpoint filename does not encode
              the expected `resume_step` / `resume_epoch`.
            - If any loaded model parameter is not in `torch.float32`.
        ValueError:
            - If the combination of `freeze_vision_backbone`, `freeze_llm_backbone`,
              and `unfreeze_last_llm_layer` does not map to a known training stage.

    Side Effects:
        - Creates directories on disk: `{run_root_dir}/{run_id}/` and `.../checkpoints/`
        - Writes `config.yaml` and `config.json` to the run directory (rank 0 only).
        - Writes dataset statistics JSON file to the run directory (rank 0 only).
        - Saves model checkpoints every `cfg.save_interval` steps.
        - Logs metrics to JSONL file and optionally to Weights & Biases.
        - Destroys the `torch.distributed` process group upon completion.

    Note:
        All filesystem writes and W&B initialization are guarded by
        `overwatch.is_rank_zero()` to prevent duplicate writes in multi-GPU runs.
        Distributed barriers (`dist.barrier()`) are used to synchronize processes
        where needed.
    """

    overwatch.info("OpenVLA Training :: Warming Up")

    # ── 1. CUDA Device Setup ──────────────────────────────────────────────────
    # `overwatch.local_rank()` returns this process's local GPU index (0 to N-1 per node).
    # Using the walrus operator `:=`, we assign it to `device_id` and simultaneously
    # pass it to `torch.cuda.set_device()`, which pins all subsequent CUDA operations
    # (tensor allocations, kernel launches) for this process to that specific GPU.
    # `empty_cache()` releases any previously allocated but unreferenced GPU memory
    # (e.g., from previous Python processes or leftover CUDA context).
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # ── 2. Run ID Construction ────────────────────────────────────────────────
    # Extract the VLA configuration identifier string (e.g., "dinosiglip-224px+mx-oxe-magic-soup-plus").
    vla_id = cfg.vla.vla_id

    # Build a descriptive run ID if the user did not provide one explicitly.
    # Format: "{vla_id}+n{num_nodes}+b{per_device_batch_size}+x{seed}"
    #   - `n{...}` = number of 8-GPU nodes (world_size / 8), encodes the compute scale
    #   - `b{...}` = per-device batch size, encodes memory usage
    #   - `x{...}` = random seed, enables distinguishing replicated experiments
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )

    # Optionally append a user-specified note suffix (e.g., "--ablation-no-augment")
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"

    # Append "--image_aug" suffix to run_id if image augmentation is enabled,
    # making it easy to distinguish augmented vs. non-augmented runs in logs/dashboards.
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # ── 3. Directory Creation & Seed Initialization ───────────────────────────
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)

    # Read the HuggingFace token from disk (if cfg.hf_token is a Path object)
    # or from an environment variable (if cfg.hf_token is a string name).
    # This token is required to download gated models (e.g., LLaMA) from HF Hub.
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    # Set the global random seed for reproducibility.
    # `get_worker_init_fn=True` additionally returns a `worker_init_fn` closure
    # that seeds each DataLoader worker process independently (using seed + worker_id),
    # preventing all workers from loading the same random data order.
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    # Create the main run directory and the nested checkpoints subdirectory.
    # The walrus operator creates `run_dir` as a Path object for reuse below.
    # `exist_ok=True` prevents errors if directories already exist (e.g., when resuming).
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # ── 4. Configuration Serialization ───────────────────────────────────────
    # Only rank 0 writes files to avoid race conditions in distributed runs.
    # draccus.dump() serializes the TrainConfig dataclass to YAML format.
    # The YAML is then re-parsed by yaml.safe_load() and immediately re-serialized
    # to JSON using json.dump(). The JSON copy is used later for HuggingFace Hub
    # model card integration (Hub expects config.json).
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))           # Save YAML config
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)                           # Parse YAML → Python dict
            json.dump(yaml_cfg, f_json, indent=2)                       # Serialize dict → JSON

    # ── 5. Model Loading ──────────────────────────────────────────────────────
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")

    if cfg.pretrained_checkpoint is not None:
        # Resuming from a prior VLA training checkpoint.
        if cfg.is_resume:
            # Validate checkpoint metadata encoded in the filename.
            # Expected filename format: "...-step-{step}-epoch-{epoch}-....pt"
            # re.search() extracts the numeric group between "step-" and "-" (or "epoch-" and "-").
            # int() converts the extracted string to an integer for comparison.
            # This is an extra sanity check to prevent accidentally resuming from the wrong checkpoint.
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        # Load the full VLA model (VLM + action tokenizer weights) from the checkpoint.
        # `load_for_training=True` ensures weights are loaded in FP32 (not quantized),
        # which is required before FSDP wraps the model and applies mixed precision.
        vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True)

    else:
        # Starting from a pretrained base VLM (no prior VLA-specific finetuning).
        # `cfg.vla.base_vlm` is either a HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
        # or a local path to a pre-downloaded model directory.
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)

    # ── 6. Model Precision Validation ────────────────────────────────────────
    # Iterate over all learnable parameters and assert each is in FP32 (torch.float32).
    # FSDP with BF16 mixed precision requires parameters to start in FP32:
    # FSDP internally casts to BF16 for forward/backward passes but maintains FP32
    # master weights for numerically stable optimizer updates.
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # ── 7. Training Stage Determination ──────────────────────────────────────
    # The "stage" string determines which model components are frozen and which
    # are trained. It is later passed to `vlm.freeze_backbones(stage)`.
    #
    # Four stages are supported, based on the combination of freeze flags:
    #   ┌────────────────────────┬──────────────────┬──────────────────────────────────────┐
    #   │ freeze_vision_backbone │ freeze_llm_backbone │ Training Stage                    │
    #   ├────────────────────────┼──────────────────┼──────────────────────────────────────┤
    #   │ False                  │ False            │ vla-full-train (full fine-tuning)      │
    #   │ True                   │ False            │ vla-train (frozen vision encoder)      │
    #   │ False                  │ True             │ vla-sandwich-train (vision + LLM head) │
    #   │ True                   │ True             │ vla-last-layer-train (LLM head only)   │
    #   └────────────────────────┴──────────────────┴──────────────────────────────────────┘

    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        # Both backbones are trainable → full fine-tuning of the entire model.
        stage = "vla-full-train"

    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        # Vision encoder is frozen; only the LLM (+ projector) is trained.
        # This is the most common setup for VLA fine-tuning: the visual features
        # are assumed to be already rich enough from pretraining (e.g., SigLIP).
        stage = "vla-train"

    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        # LLM backbone is frozen; vision encoder (+ projector + LLM last layer) are trained.
        # The `unfreeze_last_llm_layer` flag must be True; otherwise there's no trainable
        # LLM component, making the "sandwich" description inaccurate and the setup invalid.
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"

    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        # Both backbones are frozen; only the last LLM transformer layer (+ projector) is trained.
        # This is the most parameter-efficient setup, analogous to fine-tuning only the "head".
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"

    else:
        # This branch should be unreachable given boolean flags, but is kept for completeness
        # and to surface configuration errors with a descriptive message.
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # ── 8. Backbone Freezing ──────────────────────────────────────────────────
    # Call `freeze_backbones(stage)` on the VLM, which sets `requires_grad=False`
    # on the appropriate parameter groups based on the resolved `stage` string.
    # This call also logs exactly which parameter groups are frozen vs. unfrozen
    # to the Overwatch logger for debugging and reproducibility tracking.
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)

    # ── 9. Parameter Count Logging ────────────────────────────────────────────
    # Count total parameters (all tensors) and trainable parameters (requires_grad=True).
    # `.numel()` returns the number of scalar elements in a tensor.
    # Dividing by 10^6 converts to millions for readability.
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    # ── 10. Dataset & Collator Construction ───────────────────────────────────
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")

    # `get_vla_dataset_and_collator` returns three objects:
    #   - vla_dataset: A PyTorch-compatible dataset built on top of RLDS/TFDS, providing
    #                  (image, instruction, action) tuples. Supports shuffling with a large
    #                  buffer for better data mixing across episodes.
    #   - action_tokenizer: Converts continuous robot actions (joint positions, end-effector
    #                       poses, etc.) into discrete token IDs that the LLM can predict.
    #   - collator: A custom collation function that pads/stacks variable-length sequences
    #               into fixed-size batches suitable for FSDP training.
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,                                         # Root directory of OXE dataset
        cfg.vla.data_mix,                                          # Dataset mixture spec (e.g., "oxe_magic_soup_plus")
        image_transform=vlm.vision_backbone.get_image_transform(), # Preprocessing pipeline for images (resize, normalize)
        tokenizer=vlm.llm_backbone.get_tokenizer(),               # LLM tokenizer for language instructions
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,     # Function to format (instruction, action) into prompts
        default_image_resolution=vlm.vision_backbone.default_image_resolution,  # Expected input image size (e.g., 224×224)
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,          # Number of examples to hold in RAM for shuffling
        image_aug=cfg.image_aug,                                   # Whether to apply random augmentations
    )

    # ── 11. Dataset Statistics Persistence ────────────────────────────────────
    # Save per-dataset action normalization statistics (mean, std, min, max per action dim)
    # to `{run_dir}/dataset_statistics.json`. These are needed at inference time to
    # de-normalize the model's discrete token predictions back into real robot commands.
    # Only rank 0 writes to avoid concurrent file writes.
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # ── 12. Training Strategy Initialization ──────────────────────────────────
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")

    # `get_train_strategy` is a factory function that returns a strategy object
    # (e.g., `FSDPStrategy`) configured for the current training run.
    # The strategy object encapsulates:
    #   - FSDP model wrapping (sharding parameters + gradients across GPUs)
    #   - Mixed precision configuration (BF16 forward/backward, FP32 reduce)
    #   - Optimizer construction (typically AdamW)
    #   - Learning rate scheduler construction
    #   - Gradient checkpointing setup (trades compute for memory)
    #   - DataLoader construction with `worker_init_fn`
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,                                 # Strategy class name (e.g., "fsdp-full-shard")
        vlm=vlm,                                                           # The loaded VLM model (pre-FSDP wrapping)
        device_id=device_id,                                               # Local GPU index for this process
        stage=stage,                                                       # Training stage (determines FSDP wrapping rules)
        epochs=cfg.epochs,                                                 # Total training epochs
        max_steps=cfg.max_steps,                                           # Optional step cap
        global_batch_size=cfg.global_batch_size,                           # Total batch size across all GPUs
        per_device_batch_size=cfg.per_device_batch_size,                   # Per-GPU batch size
        learning_rate=cfg.learning_rate,                                   # Peak AdamW learning rate
        weight_decay=cfg.weight_decay,                                     # L2 regularization coefficient
        max_grad_norm=cfg.max_grad_norm,                                   # Gradient clipping threshold
        lr_scheduler_type=cfg.lr_scheduler_type,                           # LR schedule type ("cosine", "linear", etc.)
        warmup_ratio=cfg.warmup_ratio,                                     # Fraction of steps for LR warm-up
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,  # Trade compute for GPU memory
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,  # BF16 mixed precision
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,         # All-reduce gradients in FP32
        worker_init_fn=worker_init_fn,                                     # DataLoader worker seeding function
    )

    # Run the strategy's setup phase: wraps the model with FSDP, builds the optimizer
    # and scheduler, and calculates gradient accumulation steps based on dataset size.
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    # ── 13. Metrics Initialization ────────────────────────────────────────────
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")

    # VLAMetrics handles:
    #   - Real-time metric accumulation (loss, action accuracy, throughput, etc.)
    #   - Periodic flushing to JSONL file (one JSON object per step)
    #   - Optional W&B logging (run creation, step logging, artifact upload)
    #   - Resumption of a prior W&B run if resume_step / resume_epoch are provided
    metrics = VLAMetrics(
        cfg.trackers,          # Tuple of active tracker names: ("jsonl", "wandb")
        cfg.run_id,            # Unique run identifier (used as W&B run name and JSONL filename)
        run_dir,               # Directory where JSONL logs are written
        draccus.encode(cfg),   # Serialized config dict passed to W&B as run config
        wandb_project=cfg.wandb_project,   # W&B project name
        wandb_entity=cfg.wandb_entity,     # W&B entity (team/user)
        resume_step=cfg.resume_step,       # Step to resume W&B run from (None if starting fresh)
        resume_epoch=cfg.resume_epoch,     # Epoch to resume from (None if starting fresh)
    )

    # ── 14. Main VLA Training Loop ────────────────────────────────────────────
    overwatch.info("Starting VLA Training Loop")

    # Delegates the actual training to the strategy's `run_vla_training` method,
    # which implements the full epoch/step loop, including:
    #   - DataLoader iteration and batch transfer to GPU
    #   - Forward pass through the FSDP-wrapped VLM
    #   - Loss computation (cross-entropy over action tokens)
    #   - Backward pass with gradient accumulation
    #   - Gradient clipping and optimizer step
    #   - LR scheduler step
    #   - Metrics logging at every step
    #   - Checkpoint saving every `save_interval` steps
    train_strategy.run_vla_training(
        vla_dataset,              # The Open-X Embodiment dataset object
        collator,                 # Batch collation function
        action_tokenizer,         # Maps continuous actions ↔ token IDs
        metrics,                  # Metrics tracker for logging
        save_interval=cfg.save_interval,  # Checkpoint save frequency (in steps)
    )

    # ── 15. Finalization & Cleanup ────────────────────────────────────────────
    overwatch.info("Done with Training =>> Finalizing Metrics")

    # `metrics.finalize()` flushes any remaining buffered metrics to disk,
    # closes the JSONL file handle, and marks the W&B run as finished.
    metrics.finalize()

    overwatch.info("... and that's all, folks!")

    # `dist.barrier()` blocks all processes until every process has reached this point.
    # This ensures that rank 0 has finished writing final artifacts (e.g., last checkpoint)
    # before any process tears down the distributed context.
    dist.barrier()

    # `dist.destroy_process_group()` cleanly shuts down the NCCL/Gloo process group,
    # releasing all distributed communication resources and CUDA IPC handles.
    dist.destroy_process_group()


# ─────────────────────────────────────────────
# Script Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # When executed directly (e.g., via `torchrun ... train.py`), invoke the `train()`
    # function. The `@draccus.wrap()` decorator intercepts this call, parses CLI
    # arguments and optional YAML config files into a `TrainConfig` instance, then
    # passes it to `train(cfg)`. This pattern is analogous to HuggingFace's
    # `HfArgumentParser` or Hydra's `@hydra.main()` decorator.
    train()