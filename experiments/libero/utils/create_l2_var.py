#!/usr/bin/env python3
import re
from pathlib import Path

LIBERO_GOAL_DIR = Path("/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO/libero/libero/bddl_files/libero_goal")

LANGUAGE_VARIATIONS = {
    "open_the_middle_drawer_of_the_cabinet_syn_l2.bddl": [
        "The middle drawer of the cabinet should be opened",
        "Let the middle drawer of the cabinet be opened",
        "The middle drawer of the cabinet must be opened",
    ],

    "open_the_top_drawer_and_put_the_bowl_inside_syn_l2.bddl": [
        "The top drawer of the cabinet should be opened and the bowl should be put inside",
        "Let the top drawer of the cabinet be opened and the bowl be put inside",
        "The top drawer of the cabinet must be opened and the bowl must be put inside",
    ],

    "put_the_bowl_on_the_stove_syn_l2.bddl": [
        "The bowl should be put on the stove",
        "Let the bowl be put on the stove",
        "The bowl must be put on the stove",
    ],

    "put_the_bowl_on_the_plate_syn_l2.bddl": [
        "The bowl should be put on the plate",
        "Let the bowl be put on the plate",
        "The bowl must be put on the plate",
    ],

    "put_the_bowl_on_top_of_the_cabinet_syn_l2.bddl": [
        "The bowl should be put on the top of the cabinet",
        "Let the bowl be put on the top of the cabinet",
        "The bowl must be put on the top of the cabinet",
    ],

    "put_the_wine_bottle_on_the_rack_syn_l2.bddl": [
        "The wine bottle should be put on the rack",
        "Let the wine bottle be put on the rack",
        "The wine bottle must be put on the rack",
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet_syn_l2.bddl": [
        "The wine bottle should be put on the top of the cabinet",
        "Let the wine bottle be put on the top of the cabinet",
        "The wine bottle must be put on the top of the cabinet",
    ],

    "put_the_cream_cheese_in_the_bowl_syn_l2.bddl": [
        "The cream cheese should be put in the bowl",
        "Let the cream cheese be put in the bowl",
        "The cream cheese must be put in the bowl",
    ],

    "push_the_plate_to_the_front_of_the_stove_syn_l2.bddl": [
        "The plate should be pushed to the front of the stove",
        "Let the plate be pushed to the front of the stove",
        "The plate must be pushed to the front of the stove",
    ],

    "turn_on_the_stove_syn_l2.bddl": [
        "The stove should be turned on",
        "Let the stove be turned on",
        "The stove must be turned on",
    ],
}

LANGUAGE_PATTERN = re.compile(r"(\(:language\s+)([^)\n]+)(\))")

def replace_language(content: str, new_language: str) -> str:
    match = LANGUAGE_PATTERN.search(content)
    if not match:
        raise ValueError("Campo (:language ...) non trovato.")
    return LANGUAGE_PATTERN.sub(
        lambda m: f"{m.group(1)}{new_language}{m.group(3)}",
        content,
        count=1
    )

def main():
    if not LIBERO_GOAL_DIR.exists():
        raise FileNotFoundError(f"Cartella non trovata: {LIBERO_GOAL_DIR}")

    created = []
    missing = []

    for src_name, variants in LANGUAGE_VARIATIONS.items():
        src_path = LIBERO_GOAL_DIR / src_name

        if not src_path.exists():
            missing.append(src_name)
            print(f"[WARN] File non trovato: {src_name}")
            continue

        content = src_path.read_text(encoding="utf-8")

        for idx, new_language in enumerate(variants, start=1):
            dst_name = src_name.replace("_syn_l2.bddl", f"_syn_l2_v{idx}.bddl")
            dst_path = LIBERO_GOAL_DIR / dst_name

            new_content = replace_language(content, new_language)
            dst_path.write_text(new_content, encoding="utf-8")

            created.append(dst_name)
            print(f"[OK] {dst_name}  ->  {new_language}")

    print("\n===== SUMMARY =====")
    print(f"Created files: {len(created)}")
    if missing:
        print(f"Missing source files: {len(missing)}")
        for name in missing:
            print(f" - {name}")

if __name__ == "__main__":
    main()