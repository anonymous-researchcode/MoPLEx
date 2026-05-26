#!/usr/bin/env python
"""Prepare PERSONA pairwise and top-1 listwise datasets for DPO baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class PersonaRow:
    source_split: str
    source_index: int
    row_id: str
    persona_id: str
    persona_text: str
    preference_dimension: str
    instruction: str
    chosen: str
    rejected: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="SynthLabsAI/PERSONA")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--source_split", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--output_name_prefix",
        default="persona",
        help="Prefix for generated dataset directories, e.g. persona_50 or persona_50_other_persona.",
    )
    parser.add_argument("--instruction_column", default="instruction")
    parser.add_argument("--chosen_column", default="data")
    parser.add_argument("--rejected_column", default="original")
    parser.add_argument("--persona_column", default="persona")
    parser.add_argument("--persona_id_column", default=None)
    parser.add_argument("--num_personas", type=int, default=None)
    parser.add_argument("--persona_ids", default=None, help="Comma-separated persona ids to keep.")
    parser.add_argument("--persona_ids_file", default=None, help="One persona id per line.")
    parser.add_argument("--persona_subset_seed", type=int, default=42)
    parser.add_argument("--num_responses_per_prompt", type=int, default=4)
    parser.add_argument(
        "--eval_num_responses_per_prompt",
        type=int,
        default=None,
        help=(
            "Optional PERSONA-only eval K. When set, validation/test listwise rows use this many "
            "responses while train rows still use --num_responses_per_prompt."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        choices=("instruction_only", "persona_conditioned", "both"),
        default="both",
    )
    parser.add_argument("--hard_negative_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--split_mode",
        choices=("prompt", "row_by_persona"),
        default="prompt",
        help=(
            "How to create train/validation/test splits. 'prompt' assigns every row with the same "
            "instruction to one split; 'row_by_persona' preserves the original per-persona random split."
        ),
    )
    parser.add_argument(
        "--negative_pool",
        choices=("same_split", "all_splits"),
        default="same_split",
        help="Whether cross-persona hard negatives are sampled within each split or across all splits.",
    )
    parser.add_argument(
        "--negative_response_source",
        choices=("original", "other_persona"),
        default="original",
        help=(
            "Source for dispreferred responses. 'original' uses the generic baseline answer; "
            "'other_persona' samples personalized responses from other personas for the same instruction."
        ),
    )
    parser.add_argument(
        "--hard_negative_persona_pool",
        choices=("all", "selected"),
        default="all",
        help=(
            "Which personas may provide cross-persona hard negatives. "
            "'all' keeps the target persona subset fixed but samples negatives from the full dataset."
        ),
    )
    parser.add_argument(
        "--allow_short_listwise",
        action="store_true",
        help="Keep listwise rows with fewer than num_responses_per_prompt candidates when hard negatives are sparse.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated dataset directories under output_dir when they already exist.",
    )
    parser.add_argument("--max_source_rows", type=int, default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def stable_id(text: str, prefix: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def get_text(row: dict[str, Any], column: str) -> str:
    value = row.get(column)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def load_raw_dataset(args: argparse.Namespace) -> DatasetDict:
    load_kwargs: dict[str, Any] = {}
    if args.dataset_config is not None:
        load_kwargs["name"] = args.dataset_config
    if args.source_split is not None:
        load_kwargs["split"] = args.source_split
    if args.token:
        load_kwargs["token"] = args.token
    dataset = load_dataset(args.dataset_name, **load_kwargs)
    if isinstance(dataset, Dataset):
        return DatasetDict({args.source_split or "train": dataset})
    return dataset


def read_requested_persona_ids(args: argparse.Namespace) -> set[str] | None:
    requested: set[str] = set()
    if args.persona_ids:
        requested.update(item.strip() for item in args.persona_ids.split(",") if item.strip())
    if args.persona_ids_file:
        with Path(args.persona_ids_file).open("r", encoding="utf-8") as f:
            requested.update(line.strip() for line in f if line.strip() and not line.startswith("#"))
    return requested or None


def collect_raw_rows(raw_dataset: DatasetDict, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for source_split, split_data in raw_dataset.items():
        for source_index, row in enumerate(split_data):
            if args.max_source_rows is not None and global_index >= args.max_source_rows:
                break
            instruction = get_text(row, args.instruction_column)
            chosen = get_text(row, args.chosen_column)
            rejected = get_text(row, args.rejected_column)
            persona_text = get_text(row, args.persona_column)
            if not instruction or not chosen or not rejected or not persona_text:
                global_index += 1
                continue
            if args.persona_id_column and args.persona_id_column in row:
                persona_id = get_text(row, args.persona_id_column)
            else:
                persona_id = stable_id(persona_text, "persona")
            rows.append(
                {
                    "source_split": source_split,
                    "source_index": source_index,
                    "global_index": global_index,
                    "row_id": f"{source_split}:{source_index}",
                    "persona_id": persona_id or stable_id(persona_text, "persona"),
                    "persona_text": persona_text,
                    "instruction": instruction,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
            global_index += 1
    return rows


def select_personas(raw_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    persona_ids = sorted({row["persona_id"] for row in raw_rows})
    requested = read_requested_persona_ids(args)
    if requested is not None:
        missing = sorted(requested - set(persona_ids))
        if missing:
            raise ValueError(f"Requested persona ids not found: {missing[:10]}")
        selected = sorted(requested)
    elif args.num_personas is not None:
        if args.num_personas < 1:
            raise ValueError("--num_personas must be positive")
        if args.num_personas > len(persona_ids):
            raise ValueError(f"Requested {args.num_personas} personas but only found {len(persona_ids)}")
        rng = random.Random(args.persona_subset_seed)
        selected = sorted(rng.sample(persona_ids, args.num_personas))
    else:
        selected = persona_ids
    return selected


def split_rows_by_persona(raw_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    by_persona: dict[str, list[str]] = defaultdict(list)
    for row in raw_rows:
        by_persona[row["persona_id"]].append(row["row_id"])

    row_to_split: dict[str, str] = {}
    for persona_id, row_ids in by_persona.items():
        shuffled = list(row_ids)
        random.Random(args.split_seed + stable_int(persona_id)).shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * args.train_ratio)
        validation_end = train_end + int(n * args.validation_ratio)
        for row_id in shuffled[:train_end]:
            row_to_split[row_id] = "train"
        for row_id in shuffled[train_end:validation_end]:
            row_to_split[row_id] = "validation"
        for row_id in shuffled[validation_end:]:
            row_to_split[row_id] = "test"
    return row_to_split


def split_rows_by_prompt(raw_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    prompt_to_row_ids: dict[str, list[str]] = defaultdict(list)
    for row in raw_rows:
        prompt_to_row_ids[row["instruction"]].append(row["row_id"])

    prompts = sorted(prompt_to_row_ids)
    random.Random(args.split_seed).shuffle(prompts)
    n = len(prompts)
    train_end = int(n * args.train_ratio)
    validation_end = train_end + int(n * args.validation_ratio)

    prompt_to_split: dict[str, str] = {}
    for prompt in prompts[:train_end]:
        prompt_to_split[prompt] = "train"
    for prompt in prompts[train_end:validation_end]:
        prompt_to_split[prompt] = "validation"
    for prompt in prompts[validation_end:]:
        prompt_to_split[prompt] = "test"

    row_to_split: dict[str, str] = {}
    for prompt, row_ids in prompt_to_row_ids.items():
        split = prompt_to_split[prompt]
        for row_id in row_ids:
            row_to_split[row_id] = split
    return row_to_split


def split_rows(raw_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    split_mode = getattr(args, "split_mode", "row_by_persona")
    if split_mode == "prompt":
        return split_rows_by_prompt(raw_rows, args)
    if split_mode == "row_by_persona":
        return split_rows_by_persona(raw_rows, args)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def persona_dimension_map(selected_persona_ids: list[str]) -> dict[str, str]:
    return {persona_id: f"persona_{idx:04d}" for idx, persona_id in enumerate(selected_persona_ids)}


def build_persona_rows(
    raw_rows: list[dict[str, Any]],
    selected_persona_ids: list[str],
    args: argparse.Namespace,
    row_to_split: dict[str, str] | None = None,
    dimension_by_persona: dict[str, str] | None = None,
) -> list[PersonaRow]:
    selected = set(selected_persona_ids)
    if dimension_by_persona is None:
        dimension_by_persona = persona_dimension_map(selected_persona_ids)
    filtered_raw = [row for row in raw_rows if row["persona_id"] in selected]
    if row_to_split is None:
        row_to_split = split_rows(filtered_raw, args)
    return [
        PersonaRow(
            source_split=row["source_split"],
            source_index=int(row["source_index"]),
            row_id=row["row_id"],
            persona_id=row["persona_id"],
            persona_text=row["persona_text"],
            preference_dimension=dimension_by_persona.get(row["persona_id"], "hard_negative_pool"),
            instruction=row["instruction"],
            chosen=row["chosen"],
            rejected=row["rejected"],
            split=row_to_split[row["row_id"]],
        )
        for row in filtered_raw
    ]


def prompt_for_mode(row: PersonaRow, prompt_mode: str) -> str:
    if prompt_mode == "instruction_only":
        return row.instruction
    if prompt_mode == "persona_conditioned":
        return f"Persona:\n{row.persona_text}\n\nInstruction:\n{row.instruction}"
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def base_metadata(row: PersonaRow, prompt_mode: str) -> dict[str, Any]:
    return {
        "persona_id": row.persona_id,
        "persona_text": row.persona_text,
        "preference_dimension": row.preference_dimension,
        "instruction": row.instruction,
        "prompt_mode": prompt_mode,
        "source_split": row.source_split,
        "source_index": row.source_index,
        "source_row_id": row.row_id,
    }


def negative_response_source(args: argparse.Namespace) -> str:
    return getattr(args, "negative_response_source", "original")


def hard_negative_pool_key(args: argparse.Namespace, row: PersonaRow) -> tuple[str, str]:
    if args.negative_pool == "same_split":
        return (row.split, row.instruction)
    return ("all", row.instruction)


def build_hard_negative_pool(
    hard_negative_rows: list[PersonaRow],
    args: argparse.Namespace,
) -> dict[tuple[str, str], list[PersonaRow]]:
    prompt_pool: dict[tuple[str, str], list[PersonaRow]] = defaultdict(list)
    for row in hard_negative_rows:
        prompt_pool[hard_negative_pool_key(args, row)].append(row)
    return prompt_pool


def candidate_hard_negatives(
    row: PersonaRow,
    prompt_pool: dict[tuple[str, str], list[PersonaRow]],
    args: argparse.Namespace,
    excluded_responses: set[str] | None = None,
) -> list[PersonaRow]:
    excluded_responses = excluded_responses or set()
    candidates = []
    seen_responses = set(excluded_responses)
    for candidate in prompt_pool[hard_negative_pool_key(args, row)]:
        if candidate.persona_id == row.persona_id or not candidate.chosen:
            continue
        if candidate.chosen in seen_responses:
            continue
        candidates.append(candidate)
        seen_responses.add(candidate.chosen)
    return candidates


def sample_hard_negatives(
    row: PersonaRow,
    prompt_mode: str,
    args: argparse.Namespace,
    prompt_pool: dict[tuple[str, str], list[PersonaRow]],
    num_needed: int,
    excluded_responses: set[str] | None = None,
) -> list[PersonaRow]:
    if num_needed <= 0:
        return []
    candidates = candidate_hard_negatives(row, prompt_pool, args, excluded_responses=excluded_responses)
    rng = hard_negative_rng(args, row, prompt_mode)
    rng.shuffle(candidates)
    return candidates[:num_needed]


def build_pairwise_rows(
    rows: list[PersonaRow],
    prompt_mode: str,
    args: argparse.Namespace,
    hard_negative_rows: list[PersonaRow] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = Counter()
    source = negative_response_source(args)
    prompt_pool = None
    if source == "other_persona":
        hard_negative_rows = hard_negative_rows if hard_negative_rows is not None else rows
        prompt_pool = build_hard_negative_pool(hard_negative_rows, args)

    for row in rows:
        rejected = row.rejected
        rejected_persona_id = "generic_original"
        if source == "other_persona":
            assert prompt_pool is not None
            sampled = sample_hard_negatives(
                row,
                prompt_mode,
                args,
                prompt_pool,
                num_needed=1,
                excluded_responses={row.chosen},
            )
            if not sampled:
                stats["dropped_not_enough_hard_negatives"] += 1
                continue
            rejected = sampled[0].chosen
            rejected_persona_id = sampled[0].persona_id

        split_rows[row.split].append(
            {
                "prompt": prompt_for_mode(row, prompt_mode),
                "chosen": row.chosen,
                "rejected": rejected,
                "chosen_score": 1.0,
                "rejected_score": 0.0,
                "negative_response_source": source,
                "rejected_persona_id": rejected_persona_id,
                **base_metadata(row, prompt_mode),
            }
        )
        stats["kept"] += 1
    return split_rows, dict(stats)


def hard_negative_rng(args: argparse.Namespace, row: PersonaRow, prompt_mode: str) -> random.Random:
    seed_text = f"{args.hard_negative_seed}|{row.row_id}|{row.persona_id}|{prompt_mode}"
    return random.Random(stable_int(seed_text))


def build_listwise_rows(
    rows: list[PersonaRow],
    prompt_mode: str,
    args: argparse.Namespace,
    hard_negative_rows: list[PersonaRow] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    if args.num_responses_per_prompt < 2:
        raise ValueError("--num_responses_per_prompt must be >= 2")
    eval_k = getattr(args, "eval_num_responses_per_prompt", None)
    if eval_k is not None and eval_k < 2:
        raise ValueError("--eval_num_responses_per_prompt must be >= 2 when provided")

    hard_negative_rows = hard_negative_rows if hard_negative_rows is not None else rows
    prompt_pool = build_hard_negative_pool(hard_negative_rows, args)
    source = negative_response_source(args)
    include_original = source == "original"

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = Counter()
    for row in rows:
        row_num_responses = args.num_responses_per_prompt
        if eval_k is not None and row.split in {"validation", "test"}:
            row_num_responses = eval_k

        excluded_responses = {row.chosen}
        if include_original:
            excluded_responses.add(row.rejected)
        needed_hard_negatives = row_num_responses - 1 - int(include_original)
        candidates = candidate_hard_negatives(
            row,
            prompt_pool,
            args,
            excluded_responses=excluded_responses,
        )
        if len(candidates) < needed_hard_negatives:
            stats["dropped_not_enough_hard_negatives"] += 1
            if not args.allow_short_listwise:
                continue
        rng = hard_negative_rng(args, row, prompt_mode)
        rng.shuffle(candidates)
        sampled = candidates[:needed_hard_negatives]
        responses = [row.chosen]
        negative_persona_ids = []
        if include_original:
            responses.append(row.rejected)
            negative_persona_ids.append("generic_original")
        responses.extend(candidate.chosen for candidate in sampled)
        negative_persona_ids.extend(candidate.persona_id for candidate in sampled)
        if len(responses) < 2:
            stats["dropped_too_few_responses"] += 1
            continue
        split_rows[row.split].append(
            {
                "prompt": prompt_for_mode(row, prompt_mode),
                "responses": responses,
                "scores": [1.0] + [0.0] * (len(responses) - 1),
                "ranked_prefix_length": 1,
                "negative_response_source": source,
                "negative_persona_ids": negative_persona_ids,
                **base_metadata(row, prompt_mode),
            }
        )
        stats["kept"] += 1
        stats[f"kept_{row.split}"] += 1
    return split_rows, dict(stats)


def save_dataset(split_rows: dict[str, list[dict[str, Any]]], output_dir: Path, stats: dict[str, Any]) -> None:
    if output_dir.exists():
        if not stats.get("overwrite"):
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite or choose a new --output_dir.")
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict = DatasetDict(
        {
            split: Dataset.from_list(rows)
            for split, rows in split_rows.items()
            if rows
        }
    )
    if not dataset_dict:
        raise ValueError(f"No rows to save for {output_dir}")
    dataset_dict.save_to_disk(str(output_dir))
    stats = {
        **stats,
        "num_rows": {split: len(dataset_dict[split]) for split in dataset_dict},
    }
    with (output_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def prompt_modes(args: argparse.Namespace) -> list[str]:
    if args.prompt_mode == "both":
        return ["instruction_only", "persona_conditioned"]
    return [args.prompt_mode]


def listwise_dir_name(output_name_prefix: str, prompt_mode: str, args: argparse.Namespace) -> str:
    suffix = f"k{args.num_responses_per_prompt}"
    eval_k = getattr(args, "eval_num_responses_per_prompt", None)
    if eval_k is not None and eval_k != args.num_responses_per_prompt:
        suffix = f"{suffix}_evalk{eval_k}"
    return f"{output_name_prefix}_top1_listwise_{prompt_mode}_{suffix}"


def main() -> None:
    args = parse_args()
    raw_dataset = load_raw_dataset(args)
    raw_rows = collect_raw_rows(raw_dataset, args)
    if not raw_rows:
        raise ValueError("No valid PERSONA rows found. Check column names and source split.")

    selected_personas = select_personas(raw_rows, args)
    all_row_to_split = split_rows(raw_rows, args)
    dimension_by_persona = persona_dimension_map(selected_personas)
    persona_rows = build_persona_rows(
        raw_rows,
        selected_personas,
        args,
        row_to_split=all_row_to_split,
        dimension_by_persona=dimension_by_persona,
    )
    if not persona_rows:
        raise ValueError("No rows remained after persona selection.")

    if args.hard_negative_persona_pool == "all":
        negative_personas = sorted({row["persona_id"] for row in raw_rows})
        hard_negative_rows = build_persona_rows(
            raw_rows,
            negative_personas,
            args,
            row_to_split=all_row_to_split,
            dimension_by_persona=dimension_by_persona,
        )
    else:
        hard_negative_rows = persona_rows

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    persona_counts = Counter(row.persona_id for row in persona_rows)
    subset_payload = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "selected_persona_ids": selected_personas,
        "num_selected_personas": len(selected_personas),
        "num_rows_after_selection": len(persona_rows),
        "rows_per_persona": dict(sorted(persona_counts.items())),
        "num_responses_per_prompt": args.num_responses_per_prompt,
        "eval_num_responses_per_prompt": args.eval_num_responses_per_prompt,
        "output_name_prefix": args.output_name_prefix,
        "split_mode": args.split_mode,
        "negative_pool": args.negative_pool,
        "negative_response_source": args.negative_response_source,
        "hard_negative_persona_pool": args.hard_negative_persona_pool,
        "num_hard_negative_pool_rows": len(hard_negative_rows),
    }
    with (output_root / "persona_subset.json").open("w", encoding="utf-8") as f:
        json.dump(subset_payload, f, indent=2, sort_keys=True)
    with (output_root / f"{args.output_name_prefix}_subset.json").open("w", encoding="utf-8") as f:
        json.dump(subset_payload, f, indent=2, sort_keys=True)

    for mode in prompt_modes(args):
        pairwise_rows, pairwise_stats = build_pairwise_rows(
            persona_rows,
            mode,
            args,
            hard_negative_rows=hard_negative_rows,
        )
        pairwise_dir = output_root / f"{args.output_name_prefix}_pairwise_{mode}"
        save_dataset(
            pairwise_rows,
            pairwise_dir,
            {
                "overwrite": args.overwrite,
                "format": "pairwise",
                "prompt_mode": mode,
                "num_personas": len(selected_personas),
                "split_mode": args.split_mode,
                "negative_pool": args.negative_pool,
                "negative_response_source": args.negative_response_source,
                "hard_negative_persona_pool": args.hard_negative_persona_pool,
                "pairwise_build_stats": pairwise_stats,
            },
        )

        listwise_rows, listwise_stats = build_listwise_rows(
            persona_rows,
            mode,
            args,
            hard_negative_rows=hard_negative_rows,
        )
        listwise_dir = output_root / listwise_dir_name(args.output_name_prefix, mode, args)
        save_dataset(
            listwise_rows,
            listwise_dir,
            {
                "overwrite": args.overwrite,
                "format": "top1_listwise",
                "prompt_mode": mode,
                "num_personas": len(selected_personas),
                "num_responses_per_prompt": args.num_responses_per_prompt,
                "eval_num_responses_per_prompt": args.eval_num_responses_per_prompt,
                "ranked_prefix_length": 1,
                "split_mode": args.split_mode,
                "negative_pool": args.negative_pool,
                "negative_response_source": args.negative_response_source,
                "hard_negative_persona_pool": args.hard_negative_persona_pool,
                "listwise_build_stats": listwise_stats,
            },
        )

        print(f"Saved pairwise dataset: {pairwise_dir}")
        print(f"Saved top-1 listwise dataset: {listwise_dir}")


if __name__ == "__main__":
    main()
