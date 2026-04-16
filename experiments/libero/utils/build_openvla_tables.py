from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

TXT_GLOB = "EVAL-libero_goal-openvla-*.txt"


@dataclass
class EvalRow:
    file: str
    level: str
    seed: int
    task: int
    version: str
    original_command: str
    variation_command: str
    episodes: Optional[int]
    successes: Optional[int]
    success_rate: Optional[float]  # percent, e.g. 98.0
    is_single_task_file: bool
    mtime: float


def parse_eval_file(path: Path) -> List[EvalRow]:
    text = path.read_text(encoding="utf-8")
    level_match = re.search(r"EVALUATING:\s*(L\d)", text)
    seed_match = re.search(r"seed_(\d+)", path.name)
    if not level_match or not seed_match:
        raise ValueError(f"Impossibile leggere level/seed da {path.name}")

    level = level_match.group(1)
    seed = int(seed_match.group(1))
    is_single_task_file = "task" in path.name

    pattern = re.compile(
        r"Testing VERSION:\s*(\w+)\n"
        r"Original Command:\s*(.*?)\n"
        r"Variation Command:\s*(.*?)\n"
        r"=+\n"
        r"(.*?)(?=(?:=+\nTesting VERSION:)|\Z)",
        re.S,
    )

    rows: List[EvalRow] = []
    for match in pattern.finditer(text):
        version, original_command, variation_command, block = match.groups()

        prefix = text[: match.start()]
        task_matches = list(re.finditer(r"TASK\s+(\d+)/10", prefix))
        if not task_matches:
            raise ValueError(f"Task non trovato per {path.name} / {version}")
        task = int(task_matches[-1].group(1))

        summary_match = re.search(
            rf"VERSION {re.escape(version)} RESULTS:\s*"
            rf"Episodes:\s*(\d+)\s*"
            rf"Successes:\s*(\d+)\s*"
            rf"Success Rate:\s*([\d.]+)%",
            block,
            re.S,
        )

        if summary_match:
            episodes = int(summary_match.group(1))
            successes = int(summary_match.group(2))
            success_rate = float(summary_match.group(3))
        else:
            episodes = None
            successes = None
            success_rate = None

        rows.append(
            EvalRow(
                file=path.name,
                level=level,
                seed=seed,
                task=task,
                version=version,
                original_command=original_command,
                variation_command=variation_command,
                episodes=episodes,
                successes=successes,
                success_rate=success_rate,
                is_single_task_file=is_single_task_file,
                mtime=path.stat().st_mtime,
            )
        )

    return rows


def load_rows(input_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(input_dir.glob(TXT_GLOB)):
        for row in parse_eval_file(path):
            rows.append(row.__dict__)
    if not rows:
        raise FileNotFoundError(f"Nessun file trovato in {input_dir} con pattern {TXT_GLOB}")
    return pd.DataFrame(rows)


def choose_best_run(all_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni chiave (level, task, version, seed):
    1. preferisci una run completa (summary presente)
    2. a parità, preferisci il file single-task
    3. a parità, preferisci il file più recente
    """
    picked = []
    sort_cols = ["episodes", "is_single_task_file", "mtime"]
    for _, group in all_rows.groupby(["level", "task", "version", "seed"], dropna=False):
        group = group.sort_values(sort_cols, ascending=[False, False, False], na_position="last")
        picked.append(group.iloc[0])
    return pd.DataFrame(picked).sort_values(["level", "task", "seed", "version"]).reset_index(drop=True)


def sample_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def fmt_pct(mean: float, std: float) -> str:
    return f"{mean:.1f}% ± {std:.1f}%".replace(".", ",")


def aggregate_table(best_rows: pd.DataFrame, level: str, include_original_column: bool = False) -> pd.DataFrame:
    subset = best_rows[best_rows["level"] == level].copy()
    versions = sorted(subset["version"].dropna().unique(), key=lambda v: int(v.split("_v")[1]))

    out_rows: List[dict] = []
    for task in sorted(subset["task"].dropna().unique()):
        task_rows = subset[subset["task"] == task]
        row: Dict[str, object] = {"Task": f"Task {int(task)}"}

        mean_of_means_inputs: List[float] = []
        all_points_for_debug: List[float] = []

        for version in versions:
            vr = task_rows[task_rows["version"] == version].sort_values("seed")
            valid = vr[vr["success_rate"].notna()]
            if valid.empty:
                row[version] = "N/A"
                continue

            rates = valid["success_rate"].astype(float).tolist()
            m = float(np.mean(rates))
            s = sample_std(rates)
            row[version] = fmt_pct(m, s)
            row[f"{version}__n"] = len(rates)
            row[f"{version}__seeds"] = ",".join(map(str, valid["seed"].astype(int).tolist()))

            mean_of_means_inputs.append(m)
            all_points_for_debug.extend(rates)

        if include_original_column:
            row["Original"] = "N/A"

        if mean_of_means_inputs:
            mean_mean = float(np.mean(mean_of_means_inputs))
            std_of_version_means = sample_std(mean_of_means_inputs)
            row["Mean_of_means"] = fmt_pct(mean_mean, std_of_version_means)
            row["Mean_of_means__raw_mean"] = mean_mean
            row["Mean_of_means__raw_std"] = std_of_version_means

            # opzionale: metrica alternativa spesso più stabile
            row["Mean_all_seed_variant_points"] = fmt_pct(float(np.mean(all_points_for_debug)), sample_std(all_points_for_debug))
        else:
            row["Mean_of_means"] = "N/A"
            row["Mean_all_seed_variant_points"] = "N/A"

        out_rows.append(row)

    result = pd.DataFrame(out_rows)

    # riga OVERALL: media sulle task means
    overall: Dict[str, object] = {"Task": "OVERALL"}
    for version in versions:
        parsed = []
        for value in result.get(version, []):
            if isinstance(value, str) and value != "N/A":
                parsed.append(float(value.split("%")[0].replace(",", ".")))
        if parsed:
            overall[version] = fmt_pct(float(np.mean(parsed)), sample_std(parsed))
        else:
            overall[version] = "N/A"

    overall_means = result.get("Mean_of_means__raw_mean", pd.Series(dtype=float)).dropna().astype(float).tolist()
    if overall_means:
        overall["Mean_of_means"] = fmt_pct(float(np.mean(overall_means)), sample_std(overall_means))
    else:
        overall["Mean_of_means"] = "N/A"
    overall["Mean_all_seed_variant_points"] = "N/A"

    result = pd.concat([result, pd.DataFrame([overall])], ignore_index=True)
    return result


def build_missing_report(best_rows: pd.DataFrame) -> pd.DataFrame:
    missing = best_rows[best_rows["episodes"].isna()].copy()
    if missing.empty:
        return pd.DataFrame(columns=["level", "task", "version", "seed", "file"])
    return missing[["level", "task", "version", "seed", "file"]].sort_values(["level", "task", "seed", "version"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/openvla_tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(args.input_dir)
    best_rows = choose_best_run(raw_rows)
    missing_report = build_missing_report(best_rows)

    raw_rows.to_csv(args.output_dir / "all_parsed_runs.csv", index=False)
    best_rows.to_csv(args.output_dir / "selected_runs_after_merge.csv", index=False)
    missing_report.to_csv(args.output_dir / "missing_or_incomplete_runs.csv", index=False)

    for level in sorted(best_rows["level"].unique()):
        table = aggregate_table(best_rows, level)
        table.to_csv(args.output_dir / f"openvla_{level.lower()}_summary.csv", index=False)

    print(f"Output salvati in: {args.output_dir}")
    print("\nRun ancora mancanti/incomplete dopo il merge:")
    if missing_report.empty:
        print("  Nessuna.")
    else:
        print(missing_report.to_string(index=False))


if __name__ == "__main__":
    main()
