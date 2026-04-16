#!/usr/bin/env python3
import re
from pathlib import Path

LIBERO_GOAL_DIR = Path("/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/LIBERO/libero/libero/bddl_files/libero_goal")

LANGUAGE_VARIATIONS = {
    "open_the_middle_drawer_of_the_cabinet_syn_l1.bddl": [
        "Draw out the middle drawer of the cabinet",
        "Slide out the middle drawer of the cabinet",
        "Ease out the middle drawer of the cabinet",
    ],

    "open_the_top_drawer_and_put_the_bowl_inside_syn_l1.bddl": [
        "Draw out the top drawer of the cabinet and position the bowl inside",
        "Slide out the top drawer of the cabinet and rest the bowl inside",
        "Ease out the top drawer of the cabinet and lay the bowl inside",
    ],

    "put_the_bowl_on_the_stove_syn_l1.bddl": [
        "Position the bowl on the stove",
        "Rest the bowl on the stove",
        "Lay the bowl on the stove",
    ],

    "put_the_bowl_on_the_plate_syn_l1.bddl": [
        "Position the bowl on the plate",
        "Rest the bowl on the plate",
        "Lay the bowl on the plate",
    ],

    "put_the_bowl_on_top_of_the_cabinet_syn_l1.bddl": [
        "Position the bowl on the top of the cabinet",
        "Rest the bowl on the top of the cabinet",
        "Lay the bowl on the top of the cabinet",
    ],

    "put_the_wine_bottle_on_the_rack_syn_l1.bddl": [
        "Position the wine bottle on the rack",
        "Rest the wine bottle on the rack",
        "Lay the wine bottle on the rack",
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet_syn_l1.bddl": [
        "Position the wine bottle on the top of the cabinet",
        "Rest the wine bottle on the top of the cabinet",
        "Lay the wine bottle on the top of the cabinet",
    ],

    "put_the_cream_cheese_in_the_bowl_syn_l1.bddl": [
        "Position the cream cheese in the bowl",
        "Rest the cream cheese in the bowl",
        "Lay the cream cheese in the bowl",
    ],

    "push_the_plate_to_the_front_of_the_stove_syn_l1.bddl": [
        "Slide the plate to the front of the stove",
        "Shift the plate to the front of the stove",
        "Guide the plate to the front of the stove",
    ],

    "turn_on_the_stove_syn_l1.bddl": [
        "Ignite the stove",
        "Start the stove",
        "Power on the stove",
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
            dst_name = src_name.replace("_syn_l1.bddl", f"_syn_l1_v{idx}.bddl")
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