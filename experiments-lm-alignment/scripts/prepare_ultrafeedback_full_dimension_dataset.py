#!/usr/bin/env python
"""Prepare full UltraFeedback dimension-specific datasets for DPO/ListDPO.

The raw OpenBMB UltraFeedback split contains scored completions. This helper
turns one preference dimension into either preformatted listwise rows or
pairwise rows, then writes a deterministic train/validation/test DatasetDict
to disk. The default split is 98/1/1.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from alignment.data import _as_text, _extract_candidates


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="openbmb/UltraFeedback")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=("listwise", "pairwise"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.98)
    parser.add_argument("--validation_ratio", type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    parser.add_argument("--listwise_num_responses", type=int, default=4)
    parser.add_argument("--listwise_min_responses", type=int, default=2)
    parser.add_argument("--pairwise_strategy", choices=("all_pairs", "extreme"), default="all_pairs")
    parser.add_argument("--max_source_rows", type=int, default=None)
    return parser.parse_args()


def make_helper_args() -> SimpleNamespace:
    return SimpleNamespace(
        listwise_prompt_column="instruction",
        listwise_responses_column="completions",
        listwise_response_text_key="response",
        listwise_scores_key="scores",
        listwise_annotations_key="annotations",
    )


def build_listwise_rows(raw_dataset: Dataset, args: argparse.Namespace) -> list[dict[str, Any]]:
    helper_args = make_helper_args()
    rows: list[dict[str, Any]] = []

    for source_index, row in enumerate(raw_dataset):
        if args.max_source_rows is not None and source_index >= args.max_source_rows:
            break

        raw_prompt = row.get("instruction")
        prompt = _as_text(raw_prompt).strip()
        if not prompt:
            continue

        candidates = _extract_candidates(row, helper_args, dimension=args.dimension)
        if len(candidates) < args.listwise_min_responses:
            continue

        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        effective_k = min(args.listwise_num_responses, len(ranked))
        if effective_k < args.listwise_min_responses:
            continue

        if len(ranked) == effective_k:
            subset_rankings = [ranked]
        else:
            subset_rankings = [list(subset) for subset in combinations(ranked, effective_k)]

        for subset in subset_rankings:
            subset = sorted(subset, key=lambda item: item[1], reverse=True)
            rows.append(
                {
                    "prompt": prompt,
                    "responses": [response for response, _ in subset],
                    "scores": [float(score) for _, score in subset],
                    "preference_dimension": args.dimension,
                    "source_index": source_index,
                    "source_dataset": args.dataset_name,
                }
            )

    if not rows:
        raise ValueError(f"No listwise rows were built for dimension '{args.dimension}'.")
    return rows


def build_pairwise_rows(listwise_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in listwise_rows:
        responses = row["responses"]
        scores = row["scores"]
        n = min(len(responses), len(scores))
        if n < 2:
            continue

        pair_indices = [(0, n - 1)] if args.pairwise_strategy == "extreme" else combinations(range(n), 2)
        for chosen_idx, rejected_idx in pair_indices:
            rows.append(
                {
                    "prompt": row["prompt"],
                    "chosen": str(responses[chosen_idx]),
                    "rejected": str(responses[rejected_idx]),
                    "chosen_score": float(scores[chosen_idx]),
                    "rejected_score": float(scores[rejected_idx]),
                    "preference_dimension": row["preference_dimension"],
                    "source_index": row["source_index"],
                    "source_dataset": row["source_dataset"],
                }
            )

    if not rows:
        raise ValueError(f"No pairwise rows were built for dimension '{args.dimension}'.")
    return rows


def split_source_indices(source_indices: list[int], args: argparse.Namespace) -> dict[str, set[int]]:
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}.")

    shuffled = list(source_indices)
    random.Random(args.seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * args.train_ratio)
    validation_end = train_end + int(n * args.validation_ratio)

    return {
        "train": set(shuffled[:train_end]),
        "validation": set(shuffled[train_end:validation_end]),
        "test": set(shuffled[validation_end:]),
    }


def rows_to_dataset_dict(rows: list[dict[str, Any]], args: argparse.Namespace) -> DatasetDict:
    source_indices = sorted({int(row["source_index"]) for row in rows})
    split_indices = split_source_indices(source_indices, args)
    split_rows = {
        split: [row for row in rows if int(row["source_index"]) in indices]
        for split, indices in split_indices.items()
    }
    return DatasetDict({split: Dataset.from_list(items) for split, items in split_rows.items()})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    LOGGER.info("Loading %s split=%s", args.dataset_name, args.source_split)
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.source_split)
    listwise_rows = build_listwise_rows(raw_dataset, args)
    rows = build_pairwise_rows(listwise_rows, args) if args.format == "pairwise" else listwise_rows
    dataset_dict = rows_to_dataset_dict(rows, args)

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    stats = {
        "dataset_name": args.dataset_name,
        "source_split": args.source_split,
        "dimension": args.dimension,
        "format": args.format,
        "seed": args.seed,
        "pairwise_strategy": args.pairwise_strategy if args.format == "pairwise" else None,
        "num_rows": {split: len(dataset_dict[split]) for split in dataset_dict},
        "num_source_indices": {
            split: len(set(dataset_dict[split]["source_index"])) for split in dataset_dict
        },
    }
    with (output_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    LOGGER.info("Saved %s", output_dir)
    LOGGER.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
