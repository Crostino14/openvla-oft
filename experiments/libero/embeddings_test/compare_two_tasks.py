"""
compare_two_tasks.py

Confronto diretto tra embedding di DUE task specificati dall'utente via argparse.
Per ogni task si può specificare il nome e il livello di variazione linguistica
(default, l1, l2, l3). I due task possono anche usare scene visive diverse.

Calcola distanza coseno e distanza euclidea tra i due embedding.

Uso:
    python compare_two_tasks.py \
        --task_a put_the_bowl_on_the_stove \
        --task_b put_the_bowl_on_the_stove --level_b l1

    python compare_two_tasks.py \
        --task_a put_the_bowl_on_the_stove \
        --task_b put_the_cream_cheese_in_the_bowl --level_b l2

    # Se si vuole che entrambi i comandi siano valutati sulla stessa scena visiva:
    python compare_two_tasks.py \
        --task_a put_the_bowl_on_the_stove \
        --task_b put_the_cream_cheese_in_the_bowl \
        --scene_task put_the_bowl_on_the_stove

    # Per elencare tutti i task disponibili:
    python compare_two_tasks.py --list_tasks
"""

import argparse
import os
import re
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict

# ─────────────────── path setup ───────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
libero_root  = project_root.parent / "LIBERO"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(libero_root))

from experiments.openvla_utils import (
    get_processor,
    get_vla,
    get_action_head,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    get_libero_dummy_action,
)
from experiments.robot_utils import get_image_resize_size
from libero.libero import benchmark

# ─────────────────── constants ───────────────────
CHECKPOINT_PATH = "/home/A.CARDAMONE7/checkpoints/openvla-7b-oft-libero-goal"
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TASK_NAMES = [
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "turn_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
    "put_the_cream_cheese_in_the_bowl",
    "open_the_middle_drawer_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_stove",
]

TASK_INDEX = {name: i for i, name in enumerate(TASK_NAMES)}

VALID_LEVELS = ["default", "l1", "l2", "l3"]


# ─────────────────── BDDL command reader ───────────────────
def read_bddl_command(task_name: str, level: str = "default") -> str:
    bddl_dir = libero_root / "libero" / "libero" / "bddl_files" / "libero_goal"
    suffix = "" if level == "default" else f"_syn_{level}"
    bddl_path = bddl_dir / f"{task_name}{suffix}.bddl"
    if not bddl_path.exists():
        raise FileNotFoundError(f"File BDDL non trovato: {bddl_path}")
    text = bddl_path.read_text()
    m = re.search(r'\(:language\s+([^)]+)\)', text)
    if not m:
        raise ValueError(f"Campo :language non trovato in {bddl_path}")
    return m.group(1).strip()


# ─────────────────── config dataclass ───────────────────
@dataclass
class EmbCfg:
    pretrained_checkpoint: str = CHECKPOINT_PATH
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_goal"
    unnorm_key: str = ""
    env_img_res: int = 256
    num_steps_wait: int = 10


# ─────────────────── model loading ───────────────────
def load_model(cfg: EmbCfg):
    print(f"\nCaricamento modello da:\n  {cfg.pretrained_checkpoint}")
    model = get_vla(cfg)

    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"unnorm_key '{unnorm_key}' non trovato!"
    cfg.unnorm_key = unnorm_key

    processor = get_processor(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        try:
            proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
            print("✓ proprio_projector caricato")
        except Exception as e:
            print(f"⚠ proprio_projector non trovato: {e}")
            cfg.use_proprio = False

    action_head = None
    if cfg.use_l1_regression:
        try:
            action_head = get_action_head(cfg, model.llm_dim)
            print("✓ action_head caricato")
        except Exception as e:
            print(f"⚠ action_head non trovato: {e}")
            cfg.use_l1_regression = False

    resize_size = get_image_resize_size(cfg)
    return model, processor, action_head, proprio_projector, resize_size


# ─────────────────── embedding extraction ───────────────────
def build_prompt(task_label: str) -> str:
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def extract_prefusion_embedding(model, processor, prompt: str,
                                img_pil: Image.Image) -> np.ndarray:
    """Embedding PRE-fusione: mean-pool dei token testuali dalla lookup table,
    PRIMA del forward pass nel LLM. Puramente linguistico, nessuna influenza visiva."""
    with torch.no_grad():
        inputs = processor(prompt, img_pil).to(model.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        attn_text = inputs["attention_mask"]
        mask = attn_text.unsqueeze(-1).to(text_embeds.dtype)
        pooled = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.squeeze(0).detach().cpu().float().numpy()


def extract_embedding(model, processor, prompt: str,
                      img_pil: Image.Image,
                      wrist_pil: Optional[Image.Image] = None) -> np.ndarray:
    """Embedding POST-fusione: mean-pool dei token testuali dall'ultimo layer del LLM,
    DOPO il forward pass con i patch visivi. Rappresentazione multimodale."""
    with torch.no_grad():
        inputs = processor(prompt, img_pil).to(model.device, dtype=torch.bfloat16)

        if wrist_pil is not None:
            wrist_inputs = processor(prompt, wrist_pil).to(model.device, dtype=torch.bfloat16)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )

        patch_embeddings = model.vision_backbone(inputs["pixel_values"])
        projected_patches = model.projector(patch_embeddings)
        n_patches = projected_patches.shape[1]

        input_ids = inputs["input_ids"]
        text_embeds = model.language_model.get_input_embeddings()(input_ids)

        bos_embed = text_embeds[:, :1, :]
        text_rest = text_embeds[:, 1:, :]
        mm_embeds = torch.cat([bos_embed, projected_patches, text_rest], dim=1)

        attn_text = inputs["attention_mask"]
        patch_mask = torch.ones(
            (attn_text.shape[0], n_patches),
            dtype=attn_text.dtype, device=attn_text.device
        )
        attn_mm = torch.cat([attn_text[:, :1], patch_mask, attn_text[:, 1:]], dim=1)

        lm_out = model.language_model(
            inputs_embeds=mm_embeds,
            attention_mask=attn_mm,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = lm_out.hidden_states[-1]
        text_hidden = torch.cat([
            last_hidden[:, :1, :],
            last_hidden[:, 1 + n_patches:, :],
        ], dim=1)

        mask = attn_text.unsqueeze(-1).to(text_hidden.dtype)
        pooled = (text_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return pooled.squeeze(0).detach().cpu().float().numpy()


# ─────────────────── first frame helper ───────────────────
def get_first_frame(task, cfg: EmbCfg, resize_size, seed: int = 0):
    env, _, _ = get_libero_env(task, change_command=False, resolution=cfg.env_img_res)
    env.seed(seed)

    initial_states = None
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        task_bddl = task.bddl_file.replace(".bddl", "")
        task_id = TASK_INDEX.get(task_bddl, 0)
        initial_states = task_suite.get_task_init_states(task_id)
    except Exception:
        pass

    env.reset()
    if initial_states is not None:
        obs = env.set_init_state(initial_states[0])
    else:
        obs = env.get_observation()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    env.close()

    img_pil = Image.fromarray(
        resize_image_for_policy(img, resize_size)
    ).convert("RGB")
    wrist_pil = Image.fromarray(
        resize_image_for_policy(wrist_img, resize_size)
    ).convert("RGB")

    return img_pil, wrist_pil


# ─────────────────── distance functions ───────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ─────────────────── main ───────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Confronta embedding di due task LIBERO Goal (originali o variazioni).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Task disponibili:
{chr(10).join(f'  {i:2d}. {name}' for i, name in enumerate(TASK_NAMES))}

Livelli di variazione: default, l1, l2, l3

Esempi:
  # Confronta task originale con la sua variazione L1
  python compare_two_tasks.py \\
      --task_a put_the_bowl_on_the_stove \\
      --task_b put_the_bowl_on_the_stove --level_b l1

  # Confronta due task diversi (scene diverse)
  python compare_two_tasks.py \\
      --task_a put_the_bowl_on_the_stove \\
      --task_b put_the_cream_cheese_in_the_bowl

  # Confronta due comandi sulla STESSA scena visiva
  python compare_two_tasks.py \\
      --task_a put_the_bowl_on_the_stove \\
      --task_b put_the_cream_cheese_in_the_bowl \\
      --scene_task put_the_bowl_on_the_stove

  # Confronta variazione L2 di un task con variazione L3 di un altro
  python compare_two_tasks.py \\
      --task_a put_the_bowl_on_the_stove --level_a l2 \\
      --task_b put_the_cream_cheese_in_the_bowl --level_b l3
""",
    )
    parser.add_argument("--list_tasks", action="store_true",
                        help="Elenca i task disponibili ed esci.")
    parser.add_argument("--task_a", type=str,
                        help="Nome del primo task.")
    parser.add_argument("--level_a", type=str, default="default",
                        choices=VALID_LEVELS,
                        help="Livello di variazione per il primo task (default: default).")
    parser.add_argument("--task_b", type=str,
                        help="Nome del secondo task.")
    parser.add_argument("--level_b", type=str, default="default",
                        choices=VALID_LEVELS,
                        help="Livello di variazione per il secondo task (default: default).")
    parser.add_argument("--scene_task", type=str, default=None,
                        help="Task da cui prendere la scena visiva per ENTRAMBI i comandi. "
                             "Se non specificato, ogni task usa la propria scena.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Percorso al checkpoint del modello.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed per la simulazione (default: 0).")
    parser.add_argument("--pre_fusion", action="store_true",
                        help="Calcola ANCHE gli embedding pre-fusione (puramente testuali, "
                             "senza influenza visiva). Mostra entrambi i risultati.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Elenca task disponibili ──
    if args.list_tasks:
        print("\nTask disponibili (LIBERO Goal):")
        print("─" * 60)
        for i, name in enumerate(TASK_NAMES):
            levels = ["default"]
            for lvl in ["l1", "l2", "l3"]:
                bddl = (libero_root / "libero" / "libero" / "bddl_files" / "libero_goal"
                        / f"{name}_syn_{lvl}.bddl")
                if bddl.exists():
                    levels.append(lvl)
            print(f"  {i:2d}. {name}")
            print(f"      Livelli: {', '.join(levels)}")
        return

    # ── Validazione argomenti ──
    if not args.task_a or not args.task_b:
        print("Errore: specificare --task_a e --task_b. Usa --list_tasks per vedere i task.")
        sys.exit(1)

    if args.task_a not in TASK_INDEX:
        print(f"Errore: task_a '{args.task_a}' non trovato. Usa --list_tasks per vedere i task.")
        sys.exit(1)
    if args.task_b not in TASK_INDEX:
        print(f"Errore: task_b '{args.task_b}' non trovato. Usa --list_tasks per vedere i task.")
        sys.exit(1)
    if args.scene_task and args.scene_task not in TASK_INDEX:
        print(f"Errore: scene_task '{args.scene_task}' non trovato. Usa --list_tasks per vedere i task.")
        sys.exit(1)

    # ── Lettura comandi linguistici dai BDDL ──
    cmd_a = read_bddl_command(args.task_a, args.level_a)
    cmd_b = read_bddl_command(args.task_b, args.level_b)

    level_label_a = f"({args.level_a})" if args.level_a != "default" else "(default)"
    level_label_b = f"({args.level_b})" if args.level_b != "default" else "(default)"

    print("\n" + "=" * 90)
    print("CONFRONTO EMBEDDING - OpenVLA-OFT - LIBERO Goal")
    print("=" * 90)
    print(f"\n  Task A: {args.task_a} {level_label_a}")
    print(f"    Comando: \"{cmd_a}\"")
    print(f"\n  Task B: {args.task_b} {level_label_b}")
    print(f"    Comando: \"{cmd_b}\"")
    if args.scene_task:
        print(f"\n  Scena visiva forzata: {args.scene_task}")
    print()

    # ── Caricamento modello ──
    cfg = EmbCfg(pretrained_checkpoint=args.checkpoint)
    model, processor, action_head, proprio_projector, resize_size = load_model(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # ── Determinazione scene visive ──
    # Se scene_task è specificato, entrambi i comandi usano la stessa scena.
    # Altrimenti, ogni comando usa la scena del proprio task.
    if args.scene_task:
        scene_tasks = [args.scene_task]
    else:
        scene_tasks = list(dict.fromkeys([args.task_a, args.task_b]))

    frame_cache: Dict[str, tuple] = {}
    for task_key in scene_tasks:
        task_id = TASK_INDEX[task_key]
        task = task_suite.get_task(task_id)
        print(f"Cattura primo frame per: '{task_key}' (id={task_id})")
        img_pil, wrist_pil = get_first_frame(task, cfg, resize_size, seed=args.seed)
        frame_cache[task_key] = (img_pil, wrist_pil)

    # ── Estrazione embedding ──
    scene_key_a = args.scene_task if args.scene_task else args.task_a
    scene_key_b = args.scene_task if args.scene_task else args.task_b

    img_a, wrist_a = frame_cache[scene_key_a]
    img_b, wrist_b = frame_cache[scene_key_b]

    prompt_a = build_prompt(cmd_a)
    prompt_b = build_prompt(cmd_b)
    wrist_flag = cfg.num_images_in_input > 1

    print(f"\nEstrazione embedding POST-fusione A: \"{cmd_a}\"")
    emb_a = extract_embedding(model, processor, prompt_a, img_a,
                              wrist_a if wrist_flag else None)

    print(f"Estrazione embedding POST-fusione B: \"{cmd_b}\"")
    emb_b = extract_embedding(model, processor, prompt_b, img_b,
                              wrist_b if wrist_flag else None)

    # ── Embedding pre-fusione (se richiesto) ──
    pre_a, pre_b = None, None
    if args.pre_fusion:
        print(f"Estrazione embedding PRE-fusione A: \"{cmd_a}\"")
        pre_a = extract_prefusion_embedding(model, processor, prompt_a, img_a)
        print(f"Estrazione embedding PRE-fusione B: \"{cmd_b}\"")
        pre_b = extract_prefusion_embedding(model, processor, prompt_b, img_b)

    # ── Calcolo distanze ──
    cos_sim = cosine_similarity(emb_a, emb_b)
    euc_dist = euclidean_distance(emb_a, emb_b)

    if args.pre_fusion:
        pre_cos_sim = cosine_similarity(pre_a, pre_b)
        pre_euc_dist = euclidean_distance(pre_a, pre_b)

    # ── Stampa risultati ──
    print("\n" + "=" * 90)
    print("RISULTATI")
    print("=" * 90)

    def trunc(s, n=55):
        return s if len(s) <= n else s[:n - 1] + "…"

    print(f"\n  {'Comando A:':<14} {trunc(cmd_a, 70)}")
    print(f"  {'  Task:':<14} {args.task_a} {level_label_a}")
    print(f"  {'  Scena:':<14} {scene_key_a}")
    print()
    print(f"  {'Comando B:':<14} {trunc(cmd_b, 70)}")
    print(f"  {'  Task:':<14} {args.task_b} {level_label_b}")
    print(f"  {'  Scena:':<14} {scene_key_b}")

    print(f"\n  {'═' * 50}")
    print(f"  POST-FUSIONE (multimodale, dopo il forward LLM)")
    print(f"  {'─' * 50}")
    print(f"  Similarità coseno:  {cos_sim:.6f}")
    print(f"  Distanza euclidea:  {euc_dist:.4f}")
    print(f"  Dimensione embedding: {emb_a.shape}")
    print(f"  Norma embedding A:    {np.linalg.norm(emb_a):.4f}")
    print(f"  Norma embedding B:    {np.linalg.norm(emb_b):.4f}")

    if args.pre_fusion:
        print(f"\n  {'═' * 50}")
        print(f"  PRE-FUSIONE (puramente testuale, token lookup)")
        print(f"  {'─' * 50}")
        print(f"  Similarità coseno:  {pre_cos_sim:.6f}")
        print(f"  Distanza euclidea:  {pre_euc_dist:.4f}")
        print(f"  Dimensione embedding: {pre_a.shape}")
        print(f"  Norma embedding A:    {np.linalg.norm(pre_a):.4f}")
        print(f"  Norma embedding B:    {np.linalg.norm(pre_b):.4f}")

        print(f"\n  {'═' * 50}")
        print(f"  DELTA (post - pre fusione)")
        print(f"  {'─' * 50}")
        print(f"  Δ Similarità coseno:  {cos_sim - pre_cos_sim:+.6f}")
        print(f"  Δ Distanza euclidea:  {euc_dist - pre_euc_dist:+.4f}")

    print("\n" + "=" * 90)
    print("Fine confronto.")


if __name__ == "__main__":
    main()
