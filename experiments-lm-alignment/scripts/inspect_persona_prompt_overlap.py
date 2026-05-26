#!/usr/bin/env python
"""Inspect shared prompts across personas in SynthLabsAI/PERSONA.

This script is intentionally read-only: it loads the dataset, counts how many
distinct personas share each instruction, and estimates how many examples are
eligible for K-way hard-negative listwise construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="SynthLabsAI/PERSONA")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to load. If omitted, all available splits are inspected.",
    )
    parser.add_argument("--instruction_column", default="instruction")
    parser.add_argument("--persona_column", default="persona")
    parser.add_argument(
        "--persona_id_column",
        default=None,
        help="Optional stable persona id column. If absent, ids are derived from persona text.",
    )
    parser.add_argument(
        "--listwise_k",
        type=int,
        default=4,
        help="Candidate count for target + generic + cross-persona negatives eligibility.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional cap per split for quick inspection.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of most-shared prompts to print.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write the full summary as JSON.",
    )
    parser.add_argument(
        "--output_persona_csv",
        default=None,
        help="Optional path to write per-persona statistics as CSV.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN when set.",
    )
    return parser.parse_args()


def stable_persona_id(persona_text: str) -> str:
    digest = hashlib.sha1(persona_text.encode("utf-8")).hexdigest()[:12]
    return f"persona_{digest}"


def as_dataset_dict(dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset
    if isinstance(dataset, Dataset):
        return DatasetDict({"requested": dataset})
    if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
        raise TypeError("Streaming/iterable datasets are not supported by this inspector.")
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def get_text(row: dict[str, Any], column: str, *, row_idx: int, split: str) -> str:
    if column not in row:
        available = ", ".join(sorted(row.keys()))
        raise KeyError(f"Column '{column}' not found in split '{split}' row {row_idx}. Available: {available}")
    value = row[column]
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def summarize_split(split_name: str, split_data: Dataset, args: argparse.Namespace) -> dict[str, Any]:
    prompt_to_personas: dict[str, set[str]] = defaultdict(set)
    prompt_to_rows: Counter[str] = Counter()
    persona_to_rows: Counter[str] = Counter()
    persona_to_prompts: dict[str, set[str]] = defaultdict(set)
    persona_prompt_rows: dict[str, Counter[str]] = defaultdict(Counter)
    persona_texts: dict[str, str] = {}
    skipped_empty_prompt = 0
    skipped_empty_persona = 0
    total_rows = 0

    for row_idx, row in enumerate(split_data):
        if args.max_rows is not None and row_idx >= args.max_rows:
            break

        prompt = get_text(row, args.instruction_column, row_idx=row_idx, split=split_name)
        persona_text = get_text(row, args.persona_column, row_idx=row_idx, split=split_name)
        if not prompt:
            skipped_empty_prompt += 1
            continue
        if not persona_text:
            skipped_empty_persona += 1
            continue

        if args.persona_id_column is not None and args.persona_id_column in row:
            persona_id = str(row[args.persona_id_column]).strip()
        else:
            persona_id = stable_persona_id(persona_text)

        if not persona_id:
            persona_id = stable_persona_id(persona_text)

        if persona_id not in persona_texts:
            persona_texts[persona_id] = persona_text

        prompt_to_personas[prompt].add(persona_id)
        prompt_to_rows[prompt] += 1
        persona_to_rows[persona_id] += 1
        persona_to_prompts[persona_id].add(prompt)
        persona_prompt_rows[persona_id][prompt] += 1
        total_rows += 1

    persona_counts = [len(personas) for personas in prompt_to_personas.values()]
    row_counts = list(prompt_to_rows.values())
    shared_prompts = [prompt for prompt, personas in prompt_to_personas.items() if len(personas) >= 2]
    shared_prompt_set = set(shared_prompts)
    required_personas_for_k = max(args.listwise_k - 1, 1)
    eligible_prompts_for_k = [
        prompt for prompt, personas in prompt_to_personas.items() if len(personas) >= required_personas_for_k
    ]
    eligible_rows_for_k = sum(prompt_to_rows[prompt] for prompt in eligible_prompts_for_k)

    persona_count_histogram = Counter(persona_counts)
    row_count_histogram = Counter(row_counts)
    most_shared_prompts = sorted(
        prompt_to_personas,
        key=lambda prompt: (len(prompt_to_personas[prompt]), prompt_to_rows[prompt], prompt),
        reverse=True,
    )[: args.top_n]

    persona_stats = []
    for persona_id, row_count in persona_to_rows.items():
        prompts = persona_to_prompts.get(persona_id, set())
        shared_prompts_for_persona = prompts & shared_prompt_set
        shared_rows = sum(persona_prompt_rows[persona_id][prompt] for prompt in shared_prompts_for_persona)
        persona_stats.append(
            {
                "persona_id": persona_id,
                "persona": persona_texts.get(persona_id, ""),
                "rows": row_count,
                "distinct_prompts": len(prompts),
                "shared_distinct_prompts": len(shared_prompts_for_persona),
                "shared_rows": shared_rows,
            }
        )
    persona_stats.sort(key=lambda item: (item["rows"], item["distinct_prompts"], item["persona_id"]), reverse=True)
    
    return {
        "split": split_name,
        "total_rows_inspected": total_rows,
        "distinct_prompts": len(prompt_to_personas),
        "distinct_personas": len(persona_to_rows),
        "skipped_empty_prompt": skipped_empty_prompt,
        "skipped_empty_persona": skipped_empty_persona,
        "prompts_with_at_least_2_personas": len(shared_prompts),
        "rows_on_shared_prompts": sum(prompt_to_rows[prompt] for prompt in shared_prompts),
        "listwise_k": args.listwise_k,
        "required_distinct_personas_per_prompt_for_k": required_personas_for_k,
        "prompts_eligible_for_k": len(eligible_prompts_for_k),
        "rows_eligible_for_k": eligible_rows_for_k,
        "personas_per_prompt": {
            "min": min(persona_counts) if persona_counts else 0,
            "max": max(persona_counts) if persona_counts else 0,
            "mean": mean(persona_counts) if persona_counts else 0.0,
            "median": median(persona_counts) if persona_counts else 0.0,
        },
        "rows_per_prompt": {
            "min": min(row_counts) if row_counts else 0,
            "max": max(row_counts) if row_counts else 0,
            "mean": mean(row_counts) if row_counts else 0.0,
            "median": median(row_counts) if row_counts else 0.0,
        },
        "persona_count_histogram": dict(sorted(persona_count_histogram.items())),
        "row_count_histogram": dict(sorted(row_count_histogram.items())),
        "persona_stats": persona_stats,
        "most_shared_prompts": [
            {
                "prompt": prompt,
                "distinct_personas": len(prompt_to_personas[prompt]),
                "rows": prompt_to_rows[prompt],
            }
            for prompt in most_shared_prompts
        ],
    }


def print_split_summary(summary: dict[str, Any], top_n: int) -> None:
    print(f"\n=== Split: {summary['split']} ===")
    print(f"Rows inspected                         : {summary['total_rows_inspected']}")
    print(f"Distinct prompts                       : {summary['distinct_prompts']}")
    print(f"Distinct personas                      : {summary['distinct_personas']}")
    print(f"Prompts shared by >=2 personas         : {summary['prompts_with_at_least_2_personas']}")
    print(f"Rows on shared prompts                 : {summary['rows_on_shared_prompts']}")
    print(
        f"Prompts eligible for K={summary['listwise_k']} listwise rows : "
        f"{summary['prompts_eligible_for_k']}"
    )
    print(f"Rows eligible for K={summary['listwise_k']} listwise rows    : {summary['rows_eligible_for_k']}")
    print(f"Personas per prompt                    : {summary['personas_per_prompt']}")
    print(f"Rows per prompt                        : {summary['rows_per_prompt']}")
    print("Histogram: distinct personas per prompt")
    for count, num_prompts in summary["persona_count_histogram"].items():
        print(f"  {count}: {num_prompts}")

    if summary["most_shared_prompts"]:
        print(f"\nTop {min(top_n, len(summary['most_shared_prompts']))} most-shared prompts:")
        for item in summary["most_shared_prompts"]:
            prompt = item["prompt"].replace("\n", " ")
            if len(prompt) > 160:
                prompt = prompt[:157] + "..."
            print(f"  personas={item['distinct_personas']:>4} rows={item['rows']:>4}  {prompt}")


def main() -> None:
    args = parse_args()
    if args.listwise_k < 2:
        raise ValueError("--listwise_k must be >= 2")

    load_kwargs: dict[str, Any] = {}
    if args.dataset_config is not None:
        load_kwargs["name"] = args.dataset_config
    if args.split is not None:
        load_kwargs["split"] = args.split
    if args.token:
        load_kwargs["token"] = args.token

    dataset = load_dataset(args.dataset_name, **load_kwargs)
    dataset_dict = as_dataset_dict(dataset)

    summaries = [summarize_split(split_name, split_data, args) for split_name, split_data in dataset_dict.items()]
    for summary in summaries:
        print_split_summary(summary, args.top_n)

    if args.output_persona_csv is not None:
        output_path = Path(args.output_persona_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "split",
            "persona_id",
            "persona",
            "rows",
            "distinct_prompts",
            "shared_distinct_prompts",
            "shared_rows",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                for persona_row in summary.get("persona_stats", []):
                    writer.writerow({"split": summary["split"], **persona_row})
        print(f"\nWrote persona CSV to {output_path}")

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "instruction_column": args.instruction_column,
            "persona_column": args.persona_column,
            "persona_id_column": args.persona_id_column,
            "max_rows": args.max_rows,
            "summaries": summaries,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON summary to {output_path}")


if __name__ == "__main__":
    main()
