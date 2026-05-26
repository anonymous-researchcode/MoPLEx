#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os

from datasets import Dataset, DatasetDict, load_dataset

from alignment.cyclic_data import build_cyclic_rows, split_rows_to_dataset_dict


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create rotated-cyclic preference dataset from UltraFeedback.")
    parser.add_argument("--dataset_name", type=str, default="openbmb/UltraFeedback")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--source_split", type=str, default="train")
    parser.add_argument("--prompt_column", type=str, default="instruction")
    parser.add_argument("--responses_column", type=str, default="completions")
    parser.add_argument("--response_text_key", type=str, default="response")
    parser.add_argument("--scores_key", type=str, default="scores")
    parser.add_argument("--annotations_key", type=str, default="annotations")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["instruction_following", "honesty", "truthfulness", "helpfulness"],
        help=(
            "Ordered dimensions used for rotated filtering (m=2/3/4). "
            "Example: --dimensions instruction_following helpfulness"
        ),
    )
    parser.add_argument(
        "--dimension_pair",
        nargs=2,
        default=None,
        metavar=("DIM_A", "DIM_B"),
        help=(
            "Convenience option for exactly two dimensions. "
            "If set, it overrides --dimensions. "
            "Example: --dimension_pair instruction_following helpfulness"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cyclic_ultrafeedback_strict4d",
        help="Directory to save DatasetDict.",
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="If set, save train/validation/test (80/10/10). By default, save all rows to train only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    selected_dimensions = args.dimensions
    if args.dimension_pair is not None:
        selected_dimensions = args.dimension_pair

    logger.info("Loading source dataset: %s (config=%s, split=%s)", args.dataset_name, args.dataset_config, args.source_split)
    source = load_dataset(args.dataset_name, args.dataset_config, split=args.source_split)

    logger.info("Filtering strict cyclic rows with seed=%d and dimensions=%s", args.seed, selected_dimensions)
    rows, stats = build_cyclic_rows(
        source,
        prompt_column=args.prompt_column,
        responses_column=args.responses_column,
        response_text_key=args.response_text_key,
        scores_key=args.scores_key,
        annotations_key=args.annotations_key,
        dimensions=tuple(selected_dimensions),
        seed=args.seed,
    )

    if not rows:
        raise ValueError(
            "No rotated cyclic examples found for the selected dimensions. "
            "Try another split, different dimensions, or verify annotation keys."
        )

    if args.create_splits:
        dataset_dict = split_rows_to_dataset_dict(rows, seed=args.seed)
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
        "dimensions": list(selected_dimensions),
        "dimension_pair": list(args.dimension_pair) if args.dimension_pair is not None else None,
        "total_rows": stats.total_rows,
        "eligible_rows": stats.eligible_rows,
        "valid_score_rows": stats.valid_score_rows,
        "cycle_rows": stats.cycle_rows,
        "train_rows": len(dataset_dict["train"]),
        "validation_rows": len(dataset_dict["validation"]) if "validation" in dataset_dict else 0,
        "test_rows": len(dataset_dict["test"]) if "test" in dataset_dict else 0,
        "create_splits": args.create_splits,
    }
    metrics["eligible_ratio"] = metrics["eligible_rows"] / max(metrics["total_rows"], 1)
    metrics["valid_score_ratio"] = metrics["valid_score_rows"] / max(metrics["eligible_rows"], 1)
    metrics["cycle_ratio"] = metrics["cycle_rows"] / max(metrics["valid_score_rows"], 1)

    metrics_path = os.path.join(args.output_dir, "stats.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info("Wrote dataset stats to %s", metrics_path)


if __name__ == "__main__":
    main()