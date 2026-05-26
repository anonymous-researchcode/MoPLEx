#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os

from datasets import Dataset, DatasetDict, load_dataset

from alignment.cyclic_data import (
    build_pattern_rows,
    format_rank_pattern,
    parse_rank_pattern,
    split_rows_to_grouped_dataset_dict,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pattern-matched two-dimension dataset from UltraFeedback.")
    parser.add_argument("--dataset_name", type=str, default="openbmb/UltraFeedback")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--source_split", type=str, default="train")
    parser.add_argument("--prompt_column", type=str, default="instruction")
    parser.add_argument("--responses_column", type=str, default="completions")
    parser.add_argument("--response_text_key", type=str, default="response")
    parser.add_argument("--scores_key", type=str, default="scores")
    parser.add_argument("--annotations_key", type=str, default="annotations")
    parser.add_argument(
        "--dimension_pair",
        nargs=2,
        default=["instruction_following", "helpfulness"],
        metavar=("DIM_A", "DIM_B"),
        help="Two dimensions whose resolved rankings must match --rank_patterns.",
    )
    parser.add_argument(
        "--rank_patterns",
        nargs=2,
        default=["ABCD", "BADC"],
        metavar=("PATTERN_A", "PATTERN_B"),
        help="Two ranking patterns over A/B/C/D, e.g. ABCD or B>A>D>C.",
    )
    parser.add_argument(
        "--max_ties_per_dimension",
        type=int,
        default=1,
        help=(
            "Maximum tied score pairs allowed per selected dimension. "
            "Default 1 accepts one pair tie; use 0 for strict rankings."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ultrafeedback_pattern_2d",
        help="Directory to save DatasetDict.",
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="If set, save grouped train/validation/test (80/10/10 by source_index).",
    )
    parser.add_argument(
        "--max_source_rows",
        type=int,
        default=None,
        help="Optional cap on scanned source rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on accepted source prompts before dimension rows are emitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    dimension_pair = tuple(args.dimension_pair)
    parsed_patterns = tuple(parse_rank_pattern(pattern) for pattern in args.rank_patterns)
    normalized_patterns = [format_rank_pattern(pattern) for pattern in parsed_patterns]

    logger.info("Loading source dataset: %s (config=%s, split=%s)", args.dataset_name, args.dataset_config, args.source_split)
    source = load_dataset(args.dataset_name, args.dataset_config, split=args.source_split)

    logger.info(
        "Filtering pattern rows with seed=%d, dimensions=%s, patterns=%s, max_ties_per_dimension=%s",
        args.seed,
        dimension_pair,
        normalized_patterns,
        args.max_ties_per_dimension,
    )
    rows, stats = build_pattern_rows(
        source,
        prompt_column=args.prompt_column,
        responses_column=args.responses_column,
        response_text_key=args.response_text_key,
        scores_key=args.scores_key,
        annotations_key=args.annotations_key,
        dimension_pair=dimension_pair,
        rank_patterns=parsed_patterns,
        seed=args.seed,
        max_source_rows=args.max_source_rows,
        max_examples=args.max_examples,
        max_ties_per_dimension=args.max_ties_per_dimension,
    )

    if not rows:
        raise ValueError(
            "No pattern-matched examples found for the selected dimension pair. "
            "Try different dimensions, patterns, seed, or tie limit."
        )

    if args.create_splits:
        dataset_dict = split_rows_to_grouped_dataset_dict(rows, seed=args.seed)
    else:
        dataset_dict = DatasetDict({"train": Dataset.from_list(rows)})

    logger.info("Saving dataset to %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)

    metrics = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "source_split": args.source_split,
        "seed": args.seed,
        "dimension_pair": list(dimension_pair),
        "rank_patterns": normalized_patterns,
        "max_ties_per_dimension": args.max_ties_per_dimension,
        "total_rows": stats.total_rows,
        "eligible_rows": stats.eligible_rows,
        "valid_score_rows": stats.valid_score_rows,
        "tie_limited_rows": stats.tie_limited_rows,
        "pattern_match_source_rows": stats.pattern_match_rows,
        "emitted_listwise_rows": len(rows),
        "train_rows": len(dataset_dict["train"]),
        "validation_rows": len(dataset_dict["validation"]) if "validation" in dataset_dict else 0,
        "test_rows": len(dataset_dict["test"]) if "test" in dataset_dict else 0,
        "create_splits": args.create_splits,
        "max_source_rows": args.max_source_rows,
        "max_examples": args.max_examples,
    }
    metrics["eligible_ratio"] = metrics["eligible_rows"] / max(metrics["total_rows"], 1)
    metrics["tie_limited_ratio"] = metrics["tie_limited_rows"] / max(metrics["valid_score_rows"], 1)
    metrics["pattern_match_ratio"] = metrics["pattern_match_source_rows"] / max(metrics["tie_limited_rows"], 1)

    metrics_path = os.path.join(args.output_dir, "stats.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info("Wrote dataset stats to %s", metrics_path)


if __name__ == "__main__":
    main()
