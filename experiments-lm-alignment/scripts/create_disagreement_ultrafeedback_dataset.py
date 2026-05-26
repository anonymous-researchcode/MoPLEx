#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os

from datasets import Dataset, DatasetDict, load_dataset

from alignment.cyclic_data import build_disagreement_rows, split_rows_to_grouped_dataset_dict


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ranking-disagreement preference dataset from UltraFeedback.")
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
        help="Dimensions used for disagreement filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ultrafeedback_strict_disagreement_4d",
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
    parser.add_argument(
        "--allow_ties",
        action="store_true",
        help=(
            "Allow tied scores within a criterion. Rankings are compared as weak score orders, "
            "and tied responses are emitted adjacent using source candidate order as a deterministic tie-breaker."
        ),
    )
    parser.add_argument(
        "--max_ties_per_dimension",
        type=int,
        default=None,
        help=(
            "Optional cap on tied score pairs per criterion when --allow_ties is set. "
            "Use 1 to allow only one tied response pair among the four scores, e.g. [4, 3, 3, 1]."
        ),
    )
    parser.add_argument(
        "--min_distinct_rankings",
        type=int,
        default=2,
        help=(
            "Minimum number of distinct criterion rankings required among the selected dimensions. "
            "Use 3 to require at least three different rankings across the four UltraFeedback criteria."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    selected_dimensions = tuple(args.dimensions)

    logger.info("Loading source dataset: %s (config=%s, split=%s)", args.dataset_name, args.dataset_config, args.source_split)
    source = load_dataset(args.dataset_name, args.dataset_config, split=args.source_split)

    filter_name = "weak" if args.allow_ties else "strict"
    logger.info(
        "Filtering %s disagreement rows with seed=%d and dimensions=%s",
        filter_name,
        args.seed,
        selected_dimensions,
    )
    rows, stats = build_disagreement_rows(
        source,
        prompt_column=args.prompt_column,
        responses_column=args.responses_column,
        response_text_key=args.response_text_key,
        scores_key=args.scores_key,
        annotations_key=args.annotations_key,
        dimensions=selected_dimensions,
        seed=args.seed,
        max_source_rows=args.max_source_rows,
        max_examples=args.max_examples,
        allow_ties=args.allow_ties,
        max_ties_per_dimension=args.max_ties_per_dimension,
        min_distinct_rankings=args.min_distinct_rankings,
    )

    if not rows:
        raise ValueError(
            "No strict disagreement examples found for the selected dimensions. "
            "Try another split, different dimensions, or verify annotation keys."
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
        "dimensions": list(selected_dimensions),
        "total_rows": stats.total_rows,
        "eligible_rows": stats.eligible_rows,
        "valid_score_rows": stats.valid_score_rows,
        "valid_ranking_rows": stats.valid_ranking_rows,
        "valid_strict_ranking_rows": stats.valid_strict_ranking_rows,
        "tie_limited_rows": stats.tie_limited_rows,
        "distinct_ranking_rows": stats.distinct_ranking_rows,
        "disagreement_source_rows": stats.disagreement_rows,
        "emitted_listwise_rows": len(rows),
        "train_rows": len(dataset_dict["train"]),
        "validation_rows": len(dataset_dict["validation"]) if "validation" in dataset_dict else 0,
        "test_rows": len(dataset_dict["test"]) if "test" in dataset_dict else 0,
        "create_splits": args.create_splits,
        "max_source_rows": args.max_source_rows,
        "max_examples": args.max_examples,
        "allow_ties": args.allow_ties,
        "max_ties_per_dimension": args.max_ties_per_dimension,
        "min_distinct_rankings": args.min_distinct_rankings,
        "ranking_filter": filter_name,
    }
    metrics["eligible_ratio"] = metrics["eligible_rows"] / max(metrics["total_rows"], 1)
    metrics["strict_ranking_ratio"] = metrics["valid_strict_ranking_rows"] / max(metrics["valid_score_rows"], 1)
    metrics["ranking_ratio"] = metrics["valid_ranking_rows"] / max(metrics["valid_score_rows"], 1)
    metrics["tie_limited_ratio"] = metrics["tie_limited_rows"] / max(metrics["valid_score_rows"], 1)
    metrics["distinct_ranking_ratio"] = metrics["distinct_ranking_rows"] / max(metrics["tie_limited_rows"], 1)
    metrics["disagreement_ratio"] = metrics["disagreement_source_rows"] / max(metrics["distinct_ranking_rows"], 1)

    metrics_path = os.path.join(args.output_dir, "stats.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info("Wrote dataset stats to %s", metrics_path)


if __name__ == "__main__":
    main()
