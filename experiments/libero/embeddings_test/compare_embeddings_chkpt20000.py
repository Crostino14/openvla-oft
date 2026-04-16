"""
compare_embeddings_chkpt20000.py

Confronto tra embedding dell'input all'action model (calcolati sul primo frame)
per i comandi della suite LIBERO Goal: originale, variazione L1, L2, L3.
Usa il checkpoint 20000 di OpenVLA-OFT.

Calcola distanza coseno e distanza euclidea tra le coppie specificate.

--- SCOPO GENERALE ---
Questo script:
1. Carica il modello OpenVLA-OFT dal checkpoint 20000.
2. Per ogni gruppo di confronto, inizializza l'ambiente LIBERO corrispondente
   e cattura il PRIMO FRAME della simulazione (immagine frontale + wrist camera).
3. Per ogni comando linguistico nel gruppo, costruisce un prompt e lo passa
   attraverso il modello insieme all'immagine, estraendo l'embedding multimodale
   (hidden state dell'ultimo layer del language model, mean-pooled sui token testuali).
4. Calcola la distanza coseno e euclidea tra tutte le coppie di embedding
   nello stesso gruppo, per valutare quanto il modello distingue formulazioni
   linguistiche diverse (sinonimi, parafrasi, riferimenti indiretti).

--- OUTPUT ---
Tabelle stampate su stdout con le distanze tra coppie di comandi per ogni gruppo.
"""

import os
import re
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, Dict

# ─────────────────── path setup ───────────────────
# Risolve il percorso assoluto di QUESTO file (elimina symlink)
current_file = Path(__file__).resolve()
# Risale 3 livelli: libero/ → experiments/ → openvla-oft/ (root del progetto OpenVLA)
project_root = current_file.parent.parent.parent            # openvla-oft/
# Dalla parent di openvla-oft (robosuite_test/) entra in LIBERO/
libero_root  = project_root.parent / "LIBERO"              # robosuite_test/LIBERO
# Inserisce le root in cima a sys.path così Python trova i moduli del progetto
# con import tipo "from experiments.xxx import ..."
sys.path.insert(0, str(project_root))
# Inserisce LIBERO root per consentire "from libero.libero import benchmark"
sys.path.insert(0, str(libero_root))

# --- Import dalle utility del progetto OpenVLA-OFT ---

# get_processor: restituisce il processor del modello (tokenizer testo + preprocessor immagine).
#   Input: cfg (configurazione). Output: oggetto processor callable che accetta (prompt, image)
#   e restituisce un dict con 'input_ids', 'attention_mask', 'pixel_values'.
# get_vla: carica il modello VLA (Vision-Language-Action) completo dal checkpoint.
#   Input: cfg. Output: modello con attributi vision_backbone, projector, language_model, norm_stats, llm_dim.
# get_action_head: carica la testa di azione (MLP che predice azioni dal hidden state LLM).
#   Input: cfg, llm_dim. Output: modulo nn.Module che mappa hidden_dim → action_dim.
# get_proprio_projector: carica il proiettore propriocettivo (MLP 8-dim → llm_dim).
#   Input: cfg, llm_dim, proprio_dim. Output: nn.Module.
# resize_image_for_policy: ridimensiona un array immagine numpy alla dimensione attesa dalla policy.
#   Input: img (np.ndarray H×W×3), resize_size (int). Output: np.ndarray ridimensionato.
from experiments.openvla_utils import (
    get_processor,
    get_vla,
    get_action_head,
    get_proprio_projector,
    resize_image_for_policy,
)

# get_libero_env: crea un ambiente di simulazione LIBERO per un dato task.
#   Input: task object, change_command (bool), resolution (int).
#   Output: (env, task_description, task_bddl_path).
# get_libero_image: estrae l'immagine frontale (agentview) dall'osservazione.
#   Input: obs (dict dall'ambiente). Output: np.ndarray (H, W, 3) uint8.
# get_libero_wrist_image: estrae l'immagine della wrist camera dall'osservazione.
#   Input: obs (dict). Output: np.ndarray (H, W, 3) uint8.
# get_libero_dummy_action: genera un'azione nulla (zero) per un dato model_family.
#   Input: model_family (str). Output: np.ndarray di dimensione action_dim.
from experiments.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    get_libero_dummy_action,
)
# get_image_resize_size: calcola la dimensione di resize delle immagini per la policy.
#   Input: cfg. Output: int (es. 224 o 256, la dimensione lato dell'immagine quadrata).
from experiments.robot_utils import get_image_resize_size
# benchmark: modulo LIBERO che contiene la definizione delle task suite (goal, spatial, ecc.)
#   benchmark.get_benchmark_dict() restituisce un dict {nome_suite: classe_suite}.
from libero.libero import benchmark

# ─────────────────── constants ───────────────────

# Percorso assoluto al checkpoint 20000 del modello OpenVLA-OFT fine-tuned
# su LIBERO Goal (variante "no_noops", cioè senza azioni nulle nel dataset di training).
# Questo checkpoint contiene i pesi del VLM (vision backbone + projector + LLM)
# più eventuali action_head e proprio_projector salvati nella stessa directory.
CHECKPOINT_PATH = (
    "/home/A.CARDAMONE7/checkpoints/checkpoints_saving_folder/"
    "checkpoints_saving_folder/openvla/"
    "openvla-7b+libero_goal_no_noops_20000_chkpt"
)
# Percorso alla root di LIBERO come stringa (usato come riferimento, non direttamente nello script)
LIBERO_PATH = str(libero_root)
# Seleziona il device di computazione: GPU 0 se disponibile, altrimenti CPU.
# torch.device è un oggetto che indica dove allocare tensori e modelli.
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Mappa nome_task → indice numerico (0-based) nella suite LIBERO Goal.
# L'ordine è definito nel file tasks_info.txt del benchmark.
# Serve per recuperare lo stato iniziale corretto di ogni task
# (ogni task ha un set predefinito di configurazioni iniziali della scena).
TASK_INDEX = {
    "put_the_wine_bottle_on_top_of_the_cabinet": 0,     # task 0
    "open_the_top_drawer_and_put_the_bowl_inside": 1,    # task 1
    "turn_on_the_stove": 2,                               # task 2
    "put_the_bowl_on_top_of_the_cabinet": 3,              # task 3
    "put_the_bowl_on_the_plate": 4,                       # task 4
    "put_the_wine_bottle_on_the_rack": 5,                 # task 5
    "put_the_cream_cheese_in_the_bowl": 6,                # task 6
    "open_the_middle_drawer_of_the_cabinet": 7,           # task 7
    "push_the_plate_to_the_front_of_the_stove": 8,       # task 8
    "put_the_bowl_on_the_stove": 9,                       # task 9
}

# ─────────────────── BDDL command reader ───────────────────
def read_bddl_command(task_name: str, level: str = "default") -> str:
    """
    Legge il campo ':language' dal file BDDL per il task e il livello indicati.

    Input:
        task_name: str — nome del task (es. "put_the_wine_bottle_on_the_rack")
        level:     str — "default" | "l1" | "l2" | "l3"

    Output:
        str — testo del comando linguistico estratto dal BDDL.
    """
    bddl_dir = libero_root / "libero" / "libero" / "bddl_files" / "libero_goal"
    suffix = "" if level == "default" else f"_syn_{level}"
    bddl_path = bddl_dir / f"{task_name}{suffix}.bddl"
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")
    text = bddl_path.read_text()
    m = re.search(r'\(:language\s+([^)]+)\)', text)
    if not m:
        raise ValueError(f"No :language field found in {bddl_path}")
    return m.group(1).strip()


# ─────────────────── comparison groups ───────────────────
# Ogni gruppo definisce un confronto tra il comando default di un task e la sua
# variante linguistica (L1/L2/L3), più uno o più comandi distractor.
# I comandi vengono letti dinamicamente dai file BDDL al momento dell'esecuzione.
#
# Struttura di ogni gruppo:
#   - "level":       livello di variazione linguistica (L1=lessicale, L2=parafrasi, L3=rif. indiretto)
#   - "source_task": chiave del task LIBERO da cui si prende la SCENA VISIVA (primo frame).
#                    Tutti i comandi nel gruppo usano la STESSA immagine.
#   - "wrong_tasks": lista di nomi di task i cui comandi default sono stati effettivamente
#                    eseguiti al posto della variazione corretta durante la simulazione.
#   - "label":       etichetta leggibile per la stampa.
#   - "commands":    lista costruita a runtime in main() da read_bddl_command():
#                    [default_cmd, variant_cmd, wrong_task1_cmd, ...]
#
# La distanza viene calcolata tra TUTTE le coppie nel gruppo (triangolo superiore).
# Ci si aspetta: distanza piccola tra default e variante, distanza grande con i wrong_tasks.

COMPARISON_GROUPS = [
    # ── L1: variazioni LESSICALI semplici ──
    {
        "level": "L1",
        "source_task": "put_the_wine_bottle_on_top_of_the_cabinet",
        "wrong_tasks": ["open_the_middle_drawer_of_the_cabinet"],
        "label": "Wine bottle on top of cabinet",
    },
    {
        "level": "L1",
        "source_task": "put_the_bowl_on_top_of_the_cabinet",
        "wrong_tasks": ["put_the_bowl_on_the_stove"],
        "label": "Bowl on top of cabinet",
    },

    # ── L2: PARAFRASI strutturali ──
    {
        "level": "L2",
        "source_task": "put_the_bowl_on_top_of_the_cabinet",
        "wrong_tasks": ["put_the_bowl_on_the_stove"],
        "label": "Bowl on top of cabinet (paraphrase)",
    },
    {
        "level": "L2",
        "source_task": "put_the_cream_cheese_in_the_bowl",
        "wrong_tasks": ["put_the_bowl_on_the_stove"],
        "label": "Cream cheese in bowl (paraphrase)",
    },

    # ── L3: RIFERIMENTI INDIRETTI all'oggetto ──
    {
        "level": "L3",
        "source_task": "push_the_plate_to_the_front_of_the_stove",
        "wrong_tasks": ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove"],
        "label": "Push plate to front of stove (object ref)",
    },
    {
        "level": "L3",
        "source_task": "put_the_cream_cheese_in_the_bowl",
        "wrong_tasks": ["put_the_bowl_on_the_stove"],
        "label": "Cream cheese in bowl (object ref)",
    },
    {
        "level": "L3",
        "source_task": "put_the_wine_bottle_on_the_rack",
        "wrong_tasks": ["put_the_bowl_on_the_stove"],
        "label": "Wine bottle on rack (object ref)",
    },

    # ── ABLATION: cabinet vs drawer ──
    # Scopo: isolare l'effetto della sostituzione lessicale "cabinet" → "drawer".
    # Stessa scena visiva per tutti i comandi (wine bottle on cabinet task).
    # "raw_commands" bypassa la lettura dai BDDL: le stringhe sono hardcoded.
    {
        "level": "ABL",
        "source_task": "put_the_wine_bottle_on_top_of_the_cabinet",
        "wrong_tasks": [],
        "label": "cabinet vs drawer ablation (wine bottle)",
        "raw_commands": [
            # Training default
            "Put the wine bottle on the top of the cabinet",
            # Stesso verbo, stesso target "drawer" al posto di "cabinet"
            "Put the wine bottle on the top of the drawer",
            # L1 eval variant (verbo diverso + drawer)
            "Place the wine bottle on the top of the drawer",
            # Controfattuale: stesso verbo L1 ma con "cabinet" (NON esiste nei dati)
            "Place the wine bottle on the top of the cabinet",
            # Wrong task: unico contesto training dove "drawer" è il target dell'azione
            "Open the middle layer of the drawer",
        ],
    },
]


# ─────────────────── config dataclass ───────────────────
# Dataclass che raccoglie tutti i parametri di configurazione per il caricamento
# del modello e l'estrazione degli embedding. Usa valori di default.
@dataclass
class EmbCfg:
    # Percorso al checkpoint pre-addestrato/fine-tuned da caricare
    pretrained_checkpoint: str = CHECKPOINT_PATH
    # Famiglia del modello: "openvla" (determina il formato di input/output)
    model_family: str = "openvla"
    # Se True, usa la testa di azione con regressione L1 (predice azioni continue)
    use_l1_regression: bool = True
    # Se True, usa un modello di diffusione per predire le azioni (alternativa a L1)
    use_diffusion: bool = False
    # Numero di passi di denoising se si usa la diffusione
    num_diffusion_steps: int = 50
    # Se True, usa FiLM conditioning (Feature-wise Linear Modulation) — non usato qui
    use_film: bool = False
    # Numero di immagini in input al modello: 2 = agentview + wrist camera
    num_images_in_input: int = 2
    # Se True, include dati propriocettivi del robot (posizione joints, gripper) come input
    use_proprio: bool = True
    # Se True, applica center crop alle immagini prima del resize
    center_crop: bool = True
    # Numero di azioni predette in una singola forward pass (open-loop chunking)
    num_open_loop_steps: int = 8
    # Se True, carica il modello quantizzato a 8 bit (riduce memoria ma è meno preciso)
    load_in_8bit: bool = False
    # Se True, carica il modello quantizzato a 4 bit
    load_in_4bit: bool = False
    # Nome della task suite LIBERO da usare (determina quale benchmark caricare)
    task_suite_name: str = "libero_goal"
    # Chiave per le statistiche di normalizzazione delle azioni (viene impostata in load_model)
    unnorm_key: str = ""
    # Risoluzione delle immagini dell'ambiente LIBERO (lato in pixel)
    env_img_res: int = 256
    # Numero di passi di stabilizzazione con azioni nulle prima di catturare il frame
    num_steps_wait: int = 10


# ─────────────────── model loading ───────────────────
def load_model(cfg: EmbCfg):
    """
    Carica tutti i componenti del modello OpenVLA-OFT dal checkpoint.

    Input:
        cfg: EmbCfg — oggetto di configurazione con tutti i parametri.

    Output (tuple):
        model            — il modello VLA completo (vision_backbone + projector + language_model).
                           Ha attributi: .norm_stats (dict statistiche normalizzazione),
                           .llm_dim (int, dimensione hidden del LLM, es. 4096), .device.
        processor        — oggetto callable: processor(prompt_str, pil_image) → dict con:
                           'input_ids' (token IDs), 'attention_mask', 'pixel_values' (tensor immagine).
        action_head      — nn.Module: mappa hidden_state (llm_dim) → azione (action_dim × num_open_loop_steps).
                           Può essere None se il caricamento fallisce.
        proprio_projector — nn.Module: proietta il vettore propriocettivo (8-dim) → llm_dim.
                           Può essere None se il caricamento fallisce.
        resize_size      — int: dimensione lato a cui ridimensionare le immagini (es. 224).
    """
    print(f"\nLoading model from:\n  {cfg.pretrained_checkpoint}")
    # Carica il modello VLA dal checkpoint: include vision backbone (ViT), projector (MLP),
    # e language model (LLaMA 7B). Viene spostato su GPU automaticamente.
    # Output: oggetto modello con .vision_backbone, .projector, .language_model, .norm_stats, .llm_dim
    model = get_vla(cfg)

    # Determina la chiave per le statistiche di (de-)normalizzazione delle azioni.
    # Il modello salva le statistiche (mean/std) usate durante il training per normalizzare
    # le azioni nel range appropriato. La chiave è il nome della suite, possibilmente con "_no_noops".
    unnorm_key = cfg.task_suite_name  # "libero_goal"
    # Se "libero_goal" non è tra le chiavi ma "libero_goal_no_noops" sì, usa quest'ultima
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    # Verifica che la chiave esista, altrimenti errore fatale
    assert unnorm_key in model.norm_stats, f"unnorm_key '{unnorm_key}' not found!"
    # Salva la chiave nel cfg per uso futuro
    cfg.unnorm_key = unnorm_key

    # Carica il processor (tokenizer + image preprocessor) associato al checkpoint.
    # Output: oggetto callable che accetta (str, PIL.Image) e restituisce tensori preprocessati.
    processor = get_processor(cfg)

    # Carica il proiettore propriocettivo se richiesto dalla configurazione.
    # Questo MLP mappa il vettore propriocettivo del robot (8 dimensioni: posizioni joints + gripper)
    # nello spazio di embedding del LLM (llm_dim, es. 4096), per poterlo inserire nella sequenza.
    proprio_projector = None
    if cfg.use_proprio:
        try:
            # Input: cfg, llm_dim (int), proprio_dim=8
            # Output: nn.Module che fa Linear(8 → llm_dim)
            proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
            print("✓ proprio_projector loaded")
        except Exception as e:
            # Se il file dei pesi non esiste nel checkpoint, disabilita la propriocezione
            print(f"⚠ proprio_projector not found: {e}")
            cfg.use_proprio = False

    # Carica la testa di azione (action head) se richiesta dalla configurazione.
    # Questo MLP mappa l'hidden state del LLM → sequenza di azioni predette.
    action_head = None
    if cfg.use_l1_regression:
        try:
            # Input: cfg, llm_dim (int)
            # Output: nn.Module che fa Linear(llm_dim → action_dim * num_open_loop_steps)
            action_head = get_action_head(cfg, model.llm_dim)
            print("✓ action_head loaded")
        except Exception as e:
            print(f"⚠ action_head not found: {e}")
            cfg.use_l1_regression = False

    # Calcola la dimensione di resize delle immagini basata sulla configurazione.
    # Output: int (es. 224 per ViT-224, 384 per ViT-384)
    resize_size = get_image_resize_size(cfg)
    return model, processor, action_head, proprio_projector, resize_size


# ─────────────────── embedding extraction ───────────────────
def build_prompt(task_label: str) -> str:
    """
    Costruisce il prompt testuale nel formato atteso da OpenVLA.

    Input:
        task_label: str — descrizione del task in linguaggio naturale (es. "put the bowl on the stove").

    Output:
        str — prompt formattato: "In: What action should the robot take to {task}?\nOut:"
              Il modello è stato addestrato a generare la risposta (azione) dopo "Out:".
    """
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def find_task_token_span(processor, prompt: str, task_label: str):
    """
    Trova [start_idx, end_idx) dei token corrispondenti alla sola descrizione
    del task all'interno di input_ids (0-based, dove 0 = BOS).

    Strategia:
     1. Usa return_offsets_mapping se il tokenizer fast lo supporta (metodo robusto).
     2. Fallback: tokenizza prefisso e suffisso separatamente e usa le lunghezze.

    Input:
        processor  — processor del modello (deve avere .tokenizer o essere esso stesso)
        prompt     — stringa completa del prompt
        task_label — descrizione del task (sottostringa di prompt, già lowercased)
    Output:
        (start, end) — indici interi, slice input_ids[start:end] = token del task
    """
    tokenizer = getattr(processor, 'tokenizer', processor)
    task_lower = task_label.lower()
    task_start_char = prompt.index(task_lower)
    task_end_char   = task_start_char + len(task_lower)

    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded['offset_mapping']
        t_start, t_end = None, None
        for i, (cs, ce) in enumerate(offsets):
            if cs == 0 and ce == 0:
                continue  # BOS/EOS speciali senza span caratteri
            if t_start is None and ce > task_start_char:
                t_start = i
            if ce >= task_end_char:
                t_end = i + 1  # end esclusivo
                break
        if t_start is not None and t_end is not None:
            return t_start, t_end
    except Exception:
        pass

    # Fallback: lunghezza prefisso/suffisso tokenizzati separatamente
    PREFIX = "In: What action should the robot take to "
    SUFFIX = "?\nOut:"
    prefix_len = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    suffix_len = len(tokenizer.encode(SUFFIX, add_special_tokens=False))
    full_len   = len(tokenizer.encode(prompt,  add_special_tokens=True))
    return 1 + prefix_len, full_len - suffix_len  # 1 per il BOS


def extract_embedding(model, processor, prompt: str,
                      img_pil: Image.Image,
                      wrist_pil: Optional[Image.Image] = None,
                      task_label: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Estrae l'embedding multimodale dal modello VLA con due modalità di pooling.

    Input:
        model      — modello VLA
        processor  — processor (tokenizer + image preprocessor)
        prompt     — stringa del prompt
        img_pil    — PIL.Image agentview
        wrist_pil  — PIL.Image wrist (opzionale)
        task_label — descrizione del task (per il pooling task-only)

    Output:
        dict con due chiavi:
          "full"      — mean-pool su TUTTI i token testuali (BOS + prefisso + task + suffisso)
          "task_only" — mean-pool SOLO sui token della descrizione del task
                        (es. "put the wine bottle on the rack")
    """
    # Disabilita il calcolo dei gradienti: non serve qui e risparmia memoria GPU
    with torch.no_grad():
        # Preprocessa prompt + immagine frontale: tokenizza il testo, preprocessa l'immagine.
        # Output: dict con:
        #   'input_ids': tensor (1, seq_len_text) — IDs dei token testuali
        #   'attention_mask': tensor (1, seq_len_text) — 1 per token validi, 0 per padding
        #   'pixel_values': tensor (1, C, H, W) — immagine normalizzata per il ViT
        # .to() sposta tutto su GPU e converte in bfloat16 per efficienza
        inputs = processor(prompt, img_pil).to(model.device, dtype=torch.bfloat16)

        # Se c'è l'immagine wrist, processala separatamente e concatena i pixel_values
        if wrist_pil is not None:
            # Processa la wrist image con lo stesso prompt (il prompt serve al processor
            # per formattare correttamente, anche se i pixel_values sono indipendenti dal testo)
            wrist_inputs = processor(prompt, wrist_pil).to(model.device, dtype=torch.bfloat16)
            # Concatena i pixel values delle due immagini lungo la dimensione dei canali (dim=1)
            # Risultato: (1, 2*C, H, W) — il ViT elaborerà entrambe le immagini
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
            )

        # ── STEP 1: Vision Backbone ──
        # Passa i pixel values attraverso il Vision Transformer (ViT).
        # Input: pixel_values (1, C_total, H, W) dove C_total = C o 2*C se wrist inclusa
        # Output: patch_embeddings — tensor (1, n_patches, vision_dim)
        #   dove n_patches = (H/patch_size)² × num_images, vision_dim = dim interna del ViT
        patch_embeddings = model.vision_backbone(inputs["pixel_values"])

        # ── STEP 2: Proiezione nello spazio LLM ──
        # Il projector (tipicamente un MLP) mappa le feature visive dallo spazio del ViT
        # allo spazio di embedding del language model.
        # Input: (1, n_patches, vision_dim)
        # Output: projected_patches — (1, n_patches, llm_dim) es. (1, 576, 4096)
        projected_patches = model.projector(patch_embeddings)
        # Salva il numero di patch per sapere dove finiscono i token visivi nella sequenza
        n_patches = projected_patches.shape[1]

        # ── STEP 3: Embedding testuali ──
        # Converte gli input_ids (token IDs interi) nei corrispondenti embedding continui
        # usando la tabella di lookup del language model.
        # Input: input_ids (1, seq_len_text) — es. [1, 512, 338, 29871, ...]
        # Output: text_embeds — (1, seq_len_text, llm_dim) es. (1, 25, 4096)
        input_ids = inputs["input_ids"]
        text_embeds = model.language_model.get_input_embeddings()(input_ids)

        # ── STEP 4: Costruzione sequenza multimodale ──
        # Concatena: [BOS_token_embed] + [patch_visivi_proiettati] + [token_testuali_rimanenti]
        # Questo è il formato standard dei VLM: prima il token BOS, poi TUTTE le feature visive,
        # poi il testo. L'attenzione del transformer permette ai token testuali di "vedere"
        # le feature visive e viceversa.
        bos_embed  = text_embeds[:, :1, :]    # (1, 1, llm_dim) — primo token = BOS
        text_rest  = text_embeds[:, 1:, :]    # (1, seq_len_text-1, llm_dim) — resto del testo
        # Sequenza finale: (1, 1 + n_patches + seq_len_text - 1, llm_dim)
        mm_embeds  = torch.cat([bos_embed, projected_patches, text_rest], dim=1)

        # ── STEP 5: Costruzione attention mask multimodale ──
        # Deve corrispondere alla nuova sequenza multimodale.
        # Tutti i patch visivi sono sempre validi (mask=1), il testo mantiene la sua mask originale.
        attn_text  = inputs["attention_mask"]  # (1, seq_len_text) — mask originale del testo
        # Crea mask di tutti 1 per i patch visivi: (1, n_patches)
        patch_mask = torch.ones(
            (attn_text.shape[0], n_patches),
            dtype=attn_text.dtype, device=attn_text.device
        )
        # Concatena: [mask_BOS] + [mask_patches(tutti 1)] + [mask_testo_rimanente]
        # Risultato: (1, 1 + n_patches + seq_len_text - 1) = (1, seq_len_mm)
        attn_mm = torch.cat([attn_text[:, :1], patch_mask, attn_text[:, 1:]], dim=1)

        # ── STEP 6: Forward pass del Language Model ──
        # Passa la sequenza multimodale completa attraverso il LLM (es. LLaMA 7B).
        # Non usa input_ids ma direttamente inputs_embeds (già costruiti manualmente sopra).
        # output_hidden_states=True: chiede di restituire gli hidden states di TUTTI i layer.
        # return_dict=True: restituisce un oggetto con campi nominati.
        # Output lm_out contiene:
        #   .hidden_states: tuple di tensori, uno per layer + embedding iniziale.
        #     Ogni tensore ha shape (1, seq_len_mm, llm_dim).
        #   .logits: (1, seq_len_mm, vocab_size) — probabilità del prossimo token (non usate qui).
        lm_out = model.language_model(
            inputs_embeds=mm_embeds,
            attention_mask=attn_mm,
            output_hidden_states=True,
            return_dict=True,
        )

        # ── STEP 7: Estrazione hidden states dell'ultimo layer ──
        # hidden_states[-1] = output dell'ULTIMO layer del transformer.
        # Shape: (1, seq_len_mm, llm_dim) = (1, 1+n_patches+seq_text-1, 4096)
        # Questi sono gli hidden states "finali" che contengono la rappresentazione
        # più ricca del modello, con informazione sia visiva che linguistica miscelata.
        last_hidden = lm_out.hidden_states[-1]  # (1, seq_len_mm, hidden)

        # ── STEP 8: Isola solo i token TESTUALI ──
        # Dalla sequenza multimodale [BOS | patches | testo], estrai:
        #   - BOS (posizione 0): cattura il contesto globale
        #   - Token testuali (posizioni da 1+n_patches in poi): il prompt linguistico
        # ESCLUDE i token dei patch visivi (posizioni da 1 a n_patches).
        text_hidden = torch.cat([
            last_hidden[:, :1, :],               # BOS token: (1, 1, llm_dim)
            last_hidden[:, 1 + n_patches:, :],   # token testuali: (1, seq_text-1, llm_dim)
        ], dim=1)
        # Risultato: (1, seq_len_text, llm_dim) — stessa lunghezza della sequenza testuale originale

        # ── STEP 9a: Full pooling — mean su tutti i token testuali validi ──
        mask_full   = attn_text.unsqueeze(-1).to(text_hidden.dtype)  # (1, seq_len_text, 1)
        pooled_full = (text_hidden * mask_full).sum(dim=1) / mask_full.sum(dim=1).clamp(min=1)

        # ── STEP 9b: Task-only pooling — mean solo sui token della descrizione task ──
        if task_label is not None:
            seq_len    = text_hidden.shape[1]
            t_start, t_end = find_task_token_span(processor, prompt, task_label)
            t_start = max(0, min(t_start, seq_len))
            t_end   = max(t_start, min(t_end, seq_len))
            mask_task = torch.zeros(1, seq_len, 1,
                                    dtype=text_hidden.dtype, device=text_hidden.device)
            if t_end > t_start:
                mask_task[0, t_start:t_end, 0] = 1.0
            else:
                mask_task = mask_full  # fallback se span non trovato
            pooled_task = (text_hidden * mask_task).sum(dim=1) / mask_task.sum(dim=1).clamp(min=1)
        else:
            pooled_task = pooled_full

        # ── STEP 10: Text-only forward pass (SENZA immagine) ──
        # Passa SOLO i token testuali al LLM, senza alcun patch visivo.
        # Sequenza: [BOS | token_testuali]  →  nessuna contaminazione visiva.
        # Serve a isolare il contributo puramente linguistico del modello.
        lm_out_txt = model.language_model(
            inputs_embeds=text_embeds,        # (1, seq_len_text, llm_dim) — solo testo
            attention_mask=attn_text,         # (1, seq_len_text)
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_txt = lm_out_txt.hidden_states[-1]  # (1, seq_len_text, llm_dim)

        # Full pooling su tutti i token testuali (text-only)
        pooled_txt_full = (last_hidden_txt * mask_full).sum(dim=1) / mask_full.sum(dim=1).clamp(min=1)

        # Task-only pooling sul forward pass text-only
        if task_label is not None and t_end > t_start:
            pooled_txt_task = (last_hidden_txt * mask_task).sum(dim=1) / mask_task.sum(dim=1).clamp(min=1)
        else:
            pooled_txt_task = pooled_txt_full

    to_np = lambda t: t.squeeze(0).detach().cpu().float().numpy()
    return {
        "full":           to_np(pooled_full),
        "task_only":      to_np(pooled_task),
        "text_only_full": to_np(pooled_txt_full),
        "text_only_task": to_np(pooled_txt_task),
    }


# ─────────────────── action prediction ───────────────────
@torch.no_grad()
def extract_predicted_actions(
    model, processor, action_head, proprio_projector,
    prompt: str, img_pil, wrist_pil, cfg: EmbCfg,
) -> np.ndarray:
    """
    Replica esattamente il forward pass di predict_action e restituisce le azioni
    unnormalized predette per il comando dato.

    Output:
        np.ndarray di shape (NUM_ACTIONS_CHUNK, ACTION_DIM) = (8, 7)
        Le 7 dimensioni sono: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    # Costruisce gli input esattamente come fa get_vla_action
    inputs = processor(prompt, img_pil).to(model.device, dtype=torch.bfloat16)
    if wrist_pil is not None:
        wrist_inputs = processor(prompt, wrist_pil).to(model.device, dtype=torch.bfloat16)
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
        )

    # Propriocezione: vettore zero (primo frame prima di qualsiasi azione)
    proprio = None
    if cfg.use_proprio and proprio_projector is not None:
        proprio = np.zeros(8, dtype=np.float32)

    # Chiama predict_action del modello, che internamente:
    #  1. aggiunge 56 placeholder action token + stop token a input_ids
    #  2. azzera gli embedding degli action token
    #  3. fa il forward pass multimodale completo
    #  4. estrae gli hidden states alle posizioni degli action token
    #  5. passa action_head.predict_action() e unnormalizza
    actions, _ = model.predict_action(
        **inputs,
        unnorm_key=cfg.unnorm_key,
        proprio=proprio,
        proprio_projector=proprio_projector,
        action_head=action_head,
        use_film=cfg.use_film,
    )
    # actions: (NUM_ACTIONS_CHUNK, ACTION_DIM) = (8, 7)
    return actions  # np.ndarray float


def action_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Absolute Error tra due chunk di azioni (8,7)."""
    return float(np.abs(a - b).mean())


def action_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity tra i due chunk appiattiti (56-dim)."""
    af, bf = a.flatten(), b.flatten()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(af, bf) / denom)


def print_action_group_table(g: dict, act_cache: dict) -> None:
    """Stampa la tabella di confronto delle azioni predette per un gruppo."""
    cmds = g["commands"]
    source = g["source_task"]
    print("\n" + "─" * 100)
    print(f"[{g['level']}] {g['label']}")
    print(f"  Source task: {source}")
    print("─" * 100)

    W = 56
    print(f"{'Command A':<{W}} {'Command B':<{W}} {'MAE':>8} {'CosSim':>9}")
    print("─" * W + " " + "─" * W + " " + "─" * 8 + " " + "─" * 9)
    for i, ca in enumerate(cmds):
        for cb in cmds[i + 1:]:
            key_a = f"{source}|||{ca}"
            key_b = f"{source}|||{cb}"
            if key_a not in act_cache or key_b not in act_cache:
                continue
            act_a = act_cache[key_a]
            act_b = act_cache[key_b]
            mae  = action_mae(act_a, act_b)
            coss = action_cosine(act_a, act_b)
            ca_s = (ca[:W-1] + "…") if len(ca) > W else ca
            cb_s = (cb[:W-1] + "…") if len(cb) > W else cb
            print(f"{ca_s:<{W}} {cb_s:<{W}} {mae:>8.4f} {coss:>9.6f}")


# ─────────────────── first frame helper ───────────────────
def get_first_frame(task, cfg: EmbCfg, resize_size, seed: int = 0):
    """
    Inizializza l'ambiente LIBERO per il task dato, aspetta la stabilizzazione
    e restituisce il primo frame (agentview + wrist) come PIL Image.

    Input:
        task        — oggetto task LIBERO (ottenuto da task_suite.get_task(task_id)).
                      Contiene: .bddl_file (percorso al file di descrizione della scena),
                      .problem (descrizione del problema), .language (comando linguistico).
        cfg         — EmbCfg con i parametri di configurazione.
        resize_size — int, dimensione lato target per il resize dell'immagine.
        seed        — int, seed per la riproducibilità della simulazione.

    Output (tuple):
        img_pil   — PIL.Image (RGB) dell'immagine frontale agentview, ridimensionata.
        wrist_pil — PIL.Image (RGB) dell'immagine wrist camera, ridimensionata.
    """
    # Crea l'ambiente di simulazione MuJoCo/robosuite per questo task.
    # Input: task object, change_command=False (usa il comando originale), resolution=256
    # Output: (env, task_description_str, bddl_path_str)
    # env è un wrapper robosuite con metodi step(), reset(), seed(), close(), ecc.
    env, _, _ = get_libero_env(task, change_command=False, resolution=cfg.env_img_res)
    # Imposta il seed del generatore random dell'ambiente per riproducibilità
    env.seed(seed)

    # Tenta di caricare gli stati iniziali predefiniti dal benchmark.
    # Ogni task ha un set di configurazioni iniziali (posizioni oggetti) per garantire
    # che le valutazioni siano consistenti tra esperimenti diversi.
    initial_states = None
    try:
        # Ottiene il dizionario {nome_suite: classe_suite} dal benchmark LIBERO
        benchmark_dict = benchmark.get_benchmark_dict()
        # Istanzia la suite "libero_goal" → oggetto con metodi get_task(), get_task_init_states()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        # Estrae il nome del task dal file BDDL (rimuovendo l'estensione .bddl)
        task_bddl = task.bddl_file.replace(".bddl", "")
        # Cerca l'indice numerico nel dizionario TASK_INDEX (default 0 se non trovato)
        task_id = TASK_INDEX.get(task_bddl, 0)
        # Carica il set di stati iniziali per questo task.
        # Output: lista di np.ndarray, ognuno rappresenta uno stato MuJoCo completo
        # (posizioni, velocità, stati dei joints di tutti gli oggetti nella scena)
        initial_states = task_suite.get_task_init_states(task_id)
    except Exception:
        # Se fallisce (es. benchmark non disponibile), si usa lo stato di default di reset()
        pass

    # Resetta l'ambiente alla configurazione iniziale (posizioni oggetti, robot)
    env.reset()
    if initial_states is not None:
        # Imposta lo stato iniziale specifico del benchmark (il primo dei disponibili, indice 0)
        # Output: obs — dict di osservazioni (immagini, propriocezione, ecc.)
        obs = env.set_init_state(initial_states[0])
    else:
        # Se non ci sono stati iniziali predefiniti, usa l'osservazione post-reset
        obs = env.get_observation()

    # Passi di stabilizzazione: esegue num_steps_wait (10) azioni NULLE (zero).
    # Questo permette alla simulazione fisica di assestarsi (gli oggetti si stabilizzano
    # sulle superfici, il robot raggiunge la posizione iniziale stabile).
    # get_libero_dummy_action("openvla") restituisce un np.ndarray di zeri della dimensione corretta.
    # env.step() restituisce: (obs, reward, done, info)
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    # Estrae l'immagine frontale (agentview) dall'osservazione.
    # Input: obs dict. Output: np.ndarray (H, W, 3) uint8 — immagine RGB grezza.
    img       = get_libero_image(obs)
    # Estrae l'immagine della wrist camera (montata sul polso del robot).
    # Input: obs dict. Output: np.ndarray (H, W, 3) uint8.
    wrist_img = get_libero_wrist_image(obs)
    # Chiude l'ambiente per liberare risorse (socket MuJoCo, memoria GPU rendering)
    env.close()

    # Ridimensiona l'immagine frontale alla dimensione attesa dalla policy (es. 224×224)
    # e la converte in PIL.Image RGB.
    # resize_image_for_policy: Input (np.ndarray, int) → Output np.ndarray ridimensionato.
    # Image.fromarray: converte np.ndarray → PIL.Image.
    # .convert("RGB"): assicura il formato RGB (3 canali).
    img_pil   = Image.fromarray(
        resize_image_for_policy(img, resize_size)
    ).convert("RGB")
    # Stessa operazione per l'immagine wrist
    wrist_pil = Image.fromarray(
        resize_image_for_policy(wrist_img, resize_size)
    ).convert("RGB")

    # Restituisce le due immagini PIL pronte per il processor del modello
    return img_pil, wrist_pil


# ─────────────────── distance functions ───────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcola la similarità coseno tra due vettori.

    Formula: cos(a, b) = (a · b) / (||a|| * ||b||)
    - 1.0 = vettori identici in direzione (embedding semanticamente uguali)
    - 0.0 = vettori ortogonali (nessuna correlazione)
    - -1.0 = vettori opposti

    Input:
        a, b: np.ndarray di shape (hidden_dim,) — embedding da confrontare.
    Output:
        float — similarità coseno nell'intervallo [-1, 1].
    """
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcola la distanza euclidea (norma L2 della differenza) tra due vettori.

    Formula: d_euc(a, b) = ||a - b||_2 = sqrt(Σ(a_i - b_i)²)
    A differenza della distanza coseno, tiene conto anche della MAGNITUDINE
    dei vettori, non solo della direzione.

    Input:
        a, b: np.ndarray di shape (hidden_dim,).
    Output:
        float — distanza euclidea (≥ 0, senza limite superiore).
    """
    return float(np.linalg.norm(a - b))


# ─────────────────── pretty print table ───────────────────
def truncate(s: str, n: int = 55) -> str:
    """
    Tronca una stringa a n caratteri, aggiungendo '…' se più lunga.

    Input:
        s: str — stringa da troncare.
        n: int — lunghezza massima (default 55).
    Output:
        str — stringa originale se len ≤ n, altrimenti i primi n-1 caratteri + '…'.
    """
    return s if len(s) <= n else s[:n - 1] + "…"


def print_group_table(group: dict, embeddings: Dict[str, np.ndarray]):
    """
    Stampa una tabella formattata con le distanze coseno e euclidea
    tra tutte le coppie di comandi nello stesso gruppo.

    Input:
        group      — dict con chiavi 'level', 'label', 'source_task', 'commands'.
        embeddings — dict {comando_str: np.ndarray embedding} per ogni comando del gruppo.

    Output (su stdout):
        Tabella con header [Command A | Command B | Cosine Dist | Euclidean Dist]
        seguita da una riga per ogni coppia (i, j) con i < j (triangolo superiore).
    """
    cmds = group["commands"]  # Lista dei comandi nel gruppo
    # Stampa intestazione del gruppo con livello, etichetta e task sorgente
    print(f"\n{'─' * 100}")
    print(f"[{group['level']}] {group['label']}")
    print(f"  Source task: {group['source_task']}")
    print(f"{'─' * 100}")

    # Larghezza colonna per i comandi (56 caratteri)
    col_w = 56
    # Header della tabella con allineamento: comandi a sinistra, distanze a destra
    print(f"{'Command A':<{col_w}} {'Command B':<{col_w}} {'Cosine Sim':>12} {'Euclidean Dist':>15}")
    print(f"{'─'*col_w} {'─'*col_w} {'─'*12} {'─'*15}")

    # Crea lista di coppie (comando, embedding) nell'ordine dei comandi
    emb_list = [(c, embeddings[c]) for c in cmds]
    # Itera su tutte le coppie uniche (i, j) con i < j (triangolo superiore della matrice)
    for i in range(len(emb_list)):
        for j in range(i + 1, len(emb_list)):
            ca, ea = emb_list[i]
            cb, eb = emb_list[j]
            cs  = cosine_similarity(ea, eb)
            eud = euclidean_distance(ea, eb)
            print(f"{truncate(ca):<{col_w}} {truncate(cb):<{col_w}} {cs:>12.6f} {eud:>15.4f}")


# ─────────────────── main ───────────────────
def main():
    """
    Funzione principale. Orchestrazione completa:
    1. Carica il modello.
    2. Per ogni coppia unica (task, comando), cattura il frame e calcola l'embedding.
    3. Stampa tabelle di confronto con distanze tra coppie di embedding.
    """
    # ── STEP 1: Creazione configurazione e caricamento modello ──
    # Crea un'istanza di EmbCfg con tutti i valori di default
    cfg = EmbCfg()
    # Carica modello, processor, action_head, proprio_projector e resize_size
    # (vedi load_model per dettagli su ogni componente restituito)
    model, processor, action_head, proprio_projector, resize_size = load_model(cfg)

    # ── STEP 2: Pre-caricamento della suite di benchmark ──
    # Ottiene il dizionario delle suite benchmark disponibili → {nome: classe}
    benchmark_dict = benchmark.get_benchmark_dict()
    # Istanzia la suite "libero_goal" — contiene 10 task con scenari da cucina
    # Output: oggetto con metodi .get_task(i), .get_task_init_states(i), .n_tasks, ecc.
    task_suite     = benchmark_dict[cfg.task_suite_name]()

    # Cache per evitare di ri-inizializzare ambienti e ri-calcolare embedding.
    # frame_cache: task_key → (img_pil, wrist_pil) — 1 entry per task unico
    frame_cache: Dict[str, tuple] = {}
    # emb_cache: "task_key|||comando" → np.ndarray embedding — 1 entry per combinazione unica
    emb_cache: Dict[str, np.ndarray] = {}

    print("\n" + "=" * 100)
    print("ESTRAZIONE EMBEDDING - OpenVLA-OFT checkpoint 20000 - LIBERO Goal")
    print("=" * 100)

    # ── Risoluzione comandi dai BDDL ──
    # Popola g["commands"] leggendo le descrizioni linguistiche dai file BDDL.
    print("\n" + "─" * 60)
    print("Comandi risolti dai BDDL:")
    for g in COMPARISON_GROUPS:
        if "raw_commands" in g:
            # Gruppo con comandi hardcoded: nessuna lettura da BDDL
            g["commands"] = g["raw_commands"]
            print(f"  [{g['level']}] {g['label']}")
            for i, cmd in enumerate(g["commands"]):
                print(f"    [cmd{i:<9d}] {cmd}")
        else:
            level_str = g["level"].lower()
            default_cmd     = read_bddl_command(g["source_task"], "default")
            variant_cmd     = read_bddl_command(g["source_task"], level_str)
            wrong_cmds = [read_bddl_command(w, "default") for w in g["wrong_tasks"]]
            g["commands"] = [default_cmd, variant_cmd] + wrong_cmds
            print(f"  [{g['level']}] {g['label']}")
            tags = ["default", level_str] + [f"wrong{i+1}" for i in range(len(wrong_cmds))]
            for tag, cmd in zip(tags, g["commands"]):
                print(f"    [{tag:12s}] {cmd}")
    print("─" * 60)

    # ── STEP 3: Raccolta di tutte le coppie uniche (source_task, comando) ──
    # Scorre tutti i gruppi di confronto e tutti i comandi in ogni gruppo
    pairs = []
    for g in COMPARISON_GROUPS:
        for cmd in g["commands"]:
            # Ogni coppia: (nome_task_sorgente, stringa_comando)
            pairs.append((g["source_task"], cmd))

    # Elimina i duplicati preservando l'ordine (dict.fromkeys mantiene l'ordine di inserimento).
    # Utile perché lo stesso comando può apparire in più gruppi con lo stesso task.
    unique_pairs = list(dict.fromkeys(pairs))

    # ── STEP 4: Estrazione embedding per ogni coppia unica ──
    for source_task_key, cmd in unique_pairs:
        # Chiave univoca nel cache: "nome_task|||comando"
        # Il separatore "|||" evita ambiguità (nessun task/comando lo contiene)
        cache_key = f"{source_task_key}|||{cmd}"
        # Salta se già calcolato (deduplicazione effettiva)
        if cache_key in emb_cache:
            continue

        # Se il primo frame per questo task non è ancora in cache, lo genera
        if source_task_key not in frame_cache:
            # Trova l'indice numerico del task nel benchmark
            task_id = TASK_INDEX[source_task_key]
            # Ottiene l'oggetto task dal benchmark (contiene BDDL, lingua, ecc.)
            task    = task_suite.get_task(task_id)
            print(f"\nGetting first frame for task: '{source_task_key}' (id={task_id})")
            # Inizializza l'ambiente, stabilizza, cattura frame, chiude l'ambiente.
            # Output: (PIL.Image agentview, PIL.Image wrist) entrambe ridimensionate
            img_pil, wrist_pil = get_first_frame(task, cfg, resize_size)
            # Salva in cache per riusare con altri comandi sullo stesso task
            frame_cache[source_task_key] = (img_pil, wrist_pil)

        # Recupera le immagini dal cache
        img_pil, wrist_pil = frame_cache[source_task_key]

        # Costruisce il prompt nel formato "In: What action should the robot take to ...?\nOut:"
        prompt = build_prompt(cmd)
        print(f"  Extracting embedding: '{cmd}'")
        wrist = wrist_pil if cfg.num_images_in_input > 1 else None
        # Estrae l'embedding multimodale: restituisce dict {"full": ..., "task_only": ...}
        emb = extract_embedding(model, processor, prompt, img_pil, wrist, task_label=cmd)
        emb_cache[cache_key] = emb

    # ── STEP 4b: Predizione azioni per ogni coppia unica ──
    # Replica il forward pass di predict_action con i 56 action placeholder token.
    # Usa proprio=zero (primo frame, nessuna azione ancora eseguita).
    print("\n" + "─" * 60)
    print("Predizione azioni (action head forward pass):")
    act_cache: Dict[str, np.ndarray] = {}
    for source_task_key, cmd in unique_pairs:
        cache_key = f"{source_task_key}|||{cmd}"
        img_pil, wrist_pil = frame_cache[source_task_key]
        prompt = build_prompt(cmd)
        wrist  = wrist_pil if cfg.num_images_in_input > 1 else None
        print(f"  Predicting actions: '{cmd}'")
        actions = extract_predicted_actions(
            model, processor, action_head, proprio_projector,
            prompt, img_pil, wrist, cfg,
        )
        act_cache[cache_key] = actions
    print("─" * 60)

    # ── STEP 5: Stampa delle tabelle di confronto per tutte le modalità ──
    for mode, mode_title in [
        ("full",           "FULL TEXT POOLING       (BOS + prefisso + task + suffisso, con visione)"),
        ("task_only",      "TASK-ONLY POOLING       (solo token del task, con visione)"),
        ("text_only_full", "TEXT-ONLY FULL POOLING  (BOS + prefisso + task + suffisso, SENZA visione)"),
        ("text_only_task", "TEXT-ONLY TASK POOLING  (solo token del task, SENZA visione)"),
    ]:
        print("\n\n" + "=" * 100)
        print(f"TABELLA DI CONFRONTO EMBEDDING  — {mode_title}")
        print(f"checkpoint 20000 - OpenVLA-OFT")
        print("=" * 100)
        for g in COMPARISON_GROUPS:
            embs: Dict[str, np.ndarray] = {
                cmd: emb_cache[f"{g['source_task']}|||{cmd}"][mode]
                for cmd in g["commands"]
            }
            print_group_table(g, embs)

    # ── STEP 6: Tabella confronto azioni predette ──
    print("\n\n" + "=" * 100)
    print("TABELLA CONFRONTO AZIONI PREDETTE  — action_head forward pass (proprio=zero, primo frame)")
    print("checkpoint 20000 - OpenVLA-OFT")
    print("Metrica MAE: media del valore assoluto delle differenze sui 56 scalari (8 step × 7 dim)")
    print("Metrica CosSim: cosine similarity tra i due chunk di azioni appiattiti (56-dim)")
    print("=" * 100)
    for g in COMPARISON_GROUPS:
        print_action_group_table(g, act_cache)

    print(f"\n{'─' * 100}")
    print("Fine confronto.")

if __name__ == "__main__":
    main()
