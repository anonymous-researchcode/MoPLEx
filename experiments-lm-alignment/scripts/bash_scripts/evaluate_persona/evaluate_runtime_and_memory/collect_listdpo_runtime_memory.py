#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RUN_DIR_RE = re.compile(r"m(?P<m>\d+)-a(?P<a>\d+)-ds(?P<ds>[^-]+)-lr(?P<lr>[^-]+)-s(?P<seed>\d+)$")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[f"{prefix}_{key}"] = value
    return out


def row_for_run(run_dir: Path) -> dict[str, Any] | None:
    match = RUN_DIR_RE.search(run_dir.name)
    if match is None:
        return None

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset": run_dir.parent.name,
        "m": int(match.group("m")),
        "a": int(match.group("a")),
        "downsample": match.group("ds").replace("p", "."),
        "learning_rate": match.group("lr").replace("p", ".").replace("m", "-"),
        "seed": int(match.group("seed")),
    }
    row.update(flatten_prefixed("train", load_json(run_dir / "train_results.json")))
    row.update(flatten_prefixed("ranking_benchmark", load_json(run_dir / "ranking_benchmark_results.json")))
    row.update(flatten_prefixed("ranking_validation", load_json(run_dir / "ranking_validation_results.json")))
    row.update(flatten_prefixed("ranking_test", load_json(run_dir / "ranking_test_results.json")))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PERSONA ListDPO runtime/memory sweep metrics.")
    parser.add_argument(
        "--output_root",
        default="experiments-lm-alignment/outputs/persona/runtime-memory/listdpo-linear-approx",
        help="Root containing per-run output directories.",
    )
    parser.add_argument(
        "--output_csv",
        default="experiments-lm-alignment/outputs/persona/runtime-memory/listdpo-linear-approx-summary.csv",
        help="CSV path to write.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = []
    for train_results in sorted(output_root.glob("*/*/train_results.json")):
        row = row_for_run(train_results.parent)
        if row is not None:
            rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        print(f"No runs found under {output_root}. Wrote empty file: {output_csv}")
        return

    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "m",
        "a",
        "seed",
        "dataset",
        "learning_rate",
        "downsample",
        "train_train_wall_time_seconds",
        "train_train_gpu_peak_memory_allocated_mib",
        "train_train_gpu_peak_memory_reserved_mib",
        "ranking_benchmark_ranking_wall_time_seconds",
        "ranking_benchmark_ranking_gpu_peak_memory_allocated_mib",
        "ranking_benchmark_ranking_gpu_peak_memory_reserved_mib",
        "ranking_validation_ranking/top1_acc",
        "ranking_validation_ranking/pairwise_acc",
        "ranking_validation_mixture_posterior/top1_acc",
        "ranking_validation_mixture_posterior/pairwise_acc",
        "run_dir",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
