#!/usr/bin/env python
"""Merge pre-generated cyclic UltraFeedback listwise datasets.

The script expects DatasetDict folders produced by create_cyclic_ultrafeedback_dataset.py,
loads their train splits, deduplicates directional duplicates by default, and writes a
prompt/source-disjoint train/validation/test DatasetDict.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


EXPECTED_COLUMNS = ("prompt", "responses", "scores", "preference_dimension", "source_index")
DEFAULT_DIMENSIONS = ("instruction_following", "helpfulness", "honesty", "truthfulness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge cyclic UltraFeedback m=2 listwise datasets.")
    parser.add_argument("--input_root", type=str, default="data", help="Directory containing cyclic dataset folders.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="cyclic_ultrafeedback_m2_*",
        help="Glob pattern under input_root for dataset folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cyclic_ultrafeedback_m2_all_pairs_merged",
        help="Output directory for the merged DatasetDict.",
    )
    parser.add_argument("--validation_size", type=float, default=0.1, help="Validation source fraction or count.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test source fraction or count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for source-disjoint splitting.")
    parser.add_argument(
        "--max_rows_per_pair",
        type=int,
        default=None,
        help="Optional maximum number of rows to keep from each input folder after dimension filtering.",
    )
    parser.add_argument(
        "--allowed_dimensions",
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help="Preference dimensions to keep.",
    )
    parser.add_argument(
        "--keep_directional_duplicates",
        action="store_true",
        help="Keep duplicate rows from reverse-direction pair folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_dir if it already exists.",
    )
    return parser.parse_args()


def dataset_dirs(input_root: Path, pattern: str, output_dir: Path) -> list[Path]:
    output_resolved = output_dir.resolve()
    dirs = []
    for path in sorted(input_root.glob(pattern)):
        if not path.is_dir():
            continue
        if path.resolve() == output_resolved:
            continue
        if (path / "dataset_dict.json").exists() or (path / "dataset_info.json").exists():
            dirs.append(path)
    return dirs


def canonical_row(row: dict[str, Any], source_dataset: str) -> dict[str, Any]:
    missing = [column for column in EXPECTED_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Row from {source_dataset} is missing expected columns: {missing}")

    return {
        "prompt": str(row["prompt"]),
        "responses": [str(response) for response in row["responses"]],
        "scores": [float(score) for score in row["scores"]],
        "preference_dimension": str(row["preference_dimension"]),
        "source_index": int(row["source_index"]),
        "source_dataset": source_dataset,
    }


def dedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["source_index"],
        row["preference_dimension"],
        tuple(row["responses"]),
    )


def split_count(size: float, total: int) -> int:
    if size < 0:
        raise ValueError("Split sizes must be non-negative.")
    if size >= 1:
        count = int(size)
    else:
        count = int(round(total * size))
    return min(max(count, 0), total)


def assign_source_splits(source_indices: list[int], validation_size: float, test_size: float, seed: int) -> dict[int, str]:
    sources = list(source_indices)
    rng = random.Random(seed)
    rng.shuffle(sources)

    total = len(sources)
    test_count = split_count(test_size, total)
    remaining_after_test = total - test_count
    validation_count = split_count(validation_size, total)
    validation_count = min(validation_count, remaining_after_test)

    test_sources = set(sources[:test_count])
    validation_sources = set(sources[test_count : test_count + validation_count])

    assignments = {}
    for source_index in sources:
        if source_index in test_sources:
            assignments[source_index] = "test"
        elif source_index in validation_sources:
            assignments[source_index] = "validation"
        else:
            assignments[source_index] = "train"
    return assignments


def dimension_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["preference_dimension"] for row in rows).items()))


def validate_splits(split_rows: dict[str, list[dict[str, Any]]], check_duplicates: bool) -> None:
    all_columns = set(EXPECTED_COLUMNS) | {"source_dataset"}
    source_to_split = {}
    seen_keys = set()

    for split_name, rows in split_rows.items():
        for row in rows:
            if set(row) != all_columns:
                raise ValueError(f"Unexpected columns in {split_name}: {sorted(row)}")

            source_index = row["source_index"]
            previous_split = source_to_split.get(source_index)
            if previous_split is not None and previous_split != split_name:
                raise ValueError(
                    f"source_index={source_index} appears in both {previous_split} and {split_name}."
                )
            source_to_split[source_index] = split_name

            if check_duplicates:
                key = dedup_key(row)
                if key in seen_keys:
                    raise ValueError(f"Duplicate row remains after deduplication: {key}")
                seen_keys.add(key)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    allowed_dimensions = set(args.allowed_dimensions)

    if args.max_rows_per_pair is not None and args.max_rows_per_pair < 1:
        raise ValueError("--max_rows_per_pair must be positive when provided.")
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    input_dirs = dataset_dirs(input_root, args.pattern, output_dir)
    if not input_dirs:
        raise ValueError(f"No dataset folders found under {input_root} with pattern {args.pattern!r}.")

    rows: list[dict[str, Any]] = []
    raw_rows_loaded = 0
    rows_after_dimension_filter = 0
    per_input_stats = {}

    for input_dir in input_dirs:
        loaded = load_from_disk(str(input_dir))
        if isinstance(loaded, DatasetDict):
            if "train" not in loaded:
                raise ValueError(f"DatasetDict at {input_dir} does not contain a train split.")
            split = loaded["train"]
        elif isinstance(loaded, Dataset):
            split = loaded
        else:
            raise TypeError(f"Unsupported dataset type at {input_dir}: {type(loaded)}")

        raw_rows_loaded += len(split)
        input_rows = []
        for row in split:
            if str(row.get("preference_dimension")) not in allowed_dimensions:
                continue
            input_rows.append(canonical_row(row, source_dataset=input_dir.name))

        if args.max_rows_per_pair is not None:
            input_rows = input_rows[: args.max_rows_per_pair]

        rows_after_dimension_filter += len(input_rows)
        rows.extend(input_rows)
        per_input_stats[input_dir.name] = {
            "raw_rows": len(split),
            "kept_rows": len(input_rows),
            "dimension_counts": dimension_counts(input_rows),
        }

    duplicate_rows_removed = 0
    if not args.keep_directional_duplicates:
        deduped = []
        seen = set()
        for row in rows:
            key = dedup_key(row)
            if key in seen:
                duplicate_rows_removed += 1
                continue
            seen.add(key)
            deduped.append(row)
        rows = deduped

    if not rows:
        raise ValueError("No rows remain after filtering and deduplication.")

    source_indices = sorted({row["source_index"] for row in rows})
    assignments = assign_source_splits(
        source_indices,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    split_rows = {"train": [], "validation": [], "test": []}
    for row in rows:
        split_rows[assignments[row["source_index"]]].append(row)

    validate_splits(split_rows, check_duplicates=not args.keep_directional_duplicates)

    dataset_dict = DatasetDict(
        {
            split_name: Dataset.from_list(split_rows[split_name])
            for split_name in ("train", "validation", "test")
        }
    )

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    stats = {
        "input_root": str(input_root),
        "pattern": args.pattern,
        "input_dirs": [str(path) for path in input_dirs],
        "raw_rows_loaded": raw_rows_loaded,
        "rows_after_dimension_filter": rows_after_dimension_filter,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_dedup": len(rows),
        "num_source_indices": len(source_indices),
        "split_sizes": {split_name: len(split_rows[split_name]) for split_name in split_rows},
        "source_split_sizes": dict(sorted(Counter(assignments.values()).items())),
        "dimension_counts_by_split": {
            split_name: dimension_counts(split_rows[split_name]) for split_name in split_rows
        },
        "seed": args.seed,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "dedup_enabled": not args.keep_directional_duplicates,
        "keep_directional_duplicates": args.keep_directional_duplicates,
        "allowed_dimensions": sorted(allowed_dimensions),
        "max_rows_per_pair": args.max_rows_per_pair,
        "per_input": per_input_stats,
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Saved merged dataset to: {output_dir}")
    print(f"Wrote stats to: {stats_path}")


if __name__ == "__main__":
    main()
