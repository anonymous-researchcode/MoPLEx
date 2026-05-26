#!/usr/bin/env python
"""Augment preformatted listwise ranking datasets with generated responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {"prompt", "responses", "scores", "preference_dimension"}


@dataclass(frozen=True)
class PromptGroup:
    key: str
    prompt: str
    original_responses: tuple[str, ...]


GenerateFn = Callable[[list[str]], list[list[str]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--num_new_responses", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--augment_splits",
        nargs="+",
        default=["train"],
        help="Splits to augment with generated responses. Defaults to train only.",
    )
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--dataset_downsample_ratio",
        type=float,
        default=1.0,
        help="Downsample selected splits before generation, matching training-time dataset_downsample_ratio.",
    )
    parser.add_argument("--dataset_downsample_seed", type=int, default=None)
    parser.add_argument("--dataset_downsample_group_key", default="source_index")
    parser.add_argument(
        "--dataset_downsample_splits",
        nargs="+",
        default=["train"],
        help="Splits to downsample before augmentation. Defaults to train only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt_format", choices=("raw", "chat_auto"), default="raw")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_short_generation", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def load_ranking_dataset(dataset_name: str, dataset_config: str | None = None) -> DatasetDict:
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, dataset_config)

    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    if isinstance(dataset, DatasetDict):
        return dataset
    raise TypeError(f"Unsupported dataset object loaded from {dataset_name!r}: {type(dataset)}")


def selected_splits(dataset_dict: DatasetDict, splits: list[str] | None) -> list[str]:
    if splits is None:
        return list(dataset_dict.keys())
    missing = [split for split in splits if split not in dataset_dict]
    if missing:
        raise ValueError(f"Requested split(s) {missing} not found. Available splits: {list(dataset_dict.keys())}")
    return splits


def validate_preformatted_split(dataset: Dataset, split_name: str) -> None:
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Split {split_name!r} is missing required column(s): {sorted(missing)}")


def maybe_limit_split(dataset: Dataset, max_rows: int | None) -> Dataset:
    if max_rows is None or max_rows >= len(dataset):
        return dataset
    if max_rows < 0:
        raise ValueError("`max_rows` must be non-negative when provided.")
    return dataset.select(range(max_rows))


def split_generation_key(split: str, row: dict[str, Any]) -> str:
    source_index = row.get("source_index")
    if source_index is not None:
        return f"{split}:source_index:{source_index}"
    prompt_hash = hashlib.sha1(str(row["prompt"]).encode("utf-8")).hexdigest()
    return f"{split}:prompt_sha1:{prompt_hash}"


def generation_key(row: dict[str, Any]) -> str:
    """Backward-compatible prompt key helper for tests and external callers."""
    return split_generation_key("default", row)


def _stringify_responses(responses: list[Any]) -> list[str]:
    return [str(response) for response in responses]


def collect_prompt_groups(dataset_dict: DatasetDict, splits: list[str]) -> list[PromptGroup]:
    groups: dict[str, PromptGroup] = {}
    order: list[str] = []

    for split in splits:
        validate_preformatted_split(dataset_dict[split], split)
        for row in dataset_dict[split]:
            responses = row.get("responses")
            scores = row.get("scores")
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"Split {split!r} contains a row with an empty prompt.")
            if not isinstance(responses, list) or not isinstance(scores, list):
                raise ValueError(f"Split {split!r} contains non-list `responses` or `scores`.")
            if len(responses) != len(scores):
                raise ValueError(
                    f"Split {split!r} contains a row where responses and scores have different lengths."
                )

            key = split_generation_key(split, row)
            original_responses = tuple(_stringify_responses(responses))
            if key not in groups:
                groups[key] = PromptGroup(key=key, prompt=prompt, original_responses=original_responses)
                order.append(key)
                continue

            existing = groups[key]
            if existing.prompt != prompt:
                raise ValueError(
                    f"Generation key {key!r} maps to multiple prompts. "
                    "Use data with globally unique source_index values or remove source_index."
                )
            groups[key] = PromptGroup(
                key=key,
                prompt=existing.prompt,
                original_responses=tuple(dict.fromkeys(existing.original_responses + original_responses)),
            )

    return [groups[key] for key in order]


def should_process_split(split: str, selected: list[str] | None) -> bool:
    return selected is None or "all" in selected or split in selected


def _hashable_group_value(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def split_seed(seed: int, split_name: str) -> int:
    return seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(split_name))


def downsample_split(
    dataset: Dataset,
    *,
    split_name: str,
    ratio: float,
    seed: int,
    group_key: str,
) -> Dataset:
    if ratio <= 0 or ratio > 1:
        raise ValueError("`dataset_downsample_ratio` must be in (0, 1].")
    if len(dataset) == 0 or ratio >= 1:
        return dataset

    if group_key and group_key in dataset.column_names:
        group_values = []
        seen = set()
        for value in dataset[group_key]:
            key = _hashable_group_value(value)
            if key in seen:
                continue
            seen.add(key)
            group_values.append(key)

        keep_group_count = max(1, int(len(group_values) * ratio))
        if keep_group_count >= len(group_values):
            return dataset

        rng = random.Random(split_seed(seed, split_name))
        rng.shuffle(group_values)
        selected_groups = set(group_values[:keep_group_count])
        positions = [
            idx
            for idx, value in enumerate(dataset[group_key])
            if _hashable_group_value(value) in selected_groups
        ]
        return dataset.select(positions)

    keep_count = max(1, int(len(dataset) * ratio))
    if keep_count >= len(dataset):
        return dataset
    return dataset.shuffle(seed=split_seed(seed, split_name)).select(range(keep_count))


def prepare_output_splits(
    dataset_dict: DatasetDict,
    *,
    output_splits: list[str],
    max_rows: int | None,
    downsample_ratio: float,
    downsample_seed: int,
    downsample_group_key: str,
    downsample_splits: list[str] | None,
) -> DatasetDict:
    prepared = {}
    for split in output_splits:
        dataset = maybe_limit_split(dataset_dict[split], max_rows)
        if should_process_split(split, downsample_splits):
            dataset = downsample_split(
                dataset,
                split_name=split,
                ratio=downsample_ratio,
                seed=downsample_seed,
                group_key=downsample_group_key,
            )
        prepared[split] = dataset
    return DatasetDict(prepared)


def filter_generated_responses(
    generated: list[str],
    *,
    blocked_responses: tuple[str, ...],
    num_new_responses: int,
    deduplicate: bool,
    allow_short_generation: bool,
) -> tuple[list[str], int]:
    kept: list[str] = []
    blocked = {response.strip() for response in blocked_responses}
    dropped = 0

    for response in generated:
        text = str(response).strip()
        if not text:
            dropped += 1
            continue
        if deduplicate and (text in blocked or text in kept):
            dropped += 1
            continue
        kept.append(text)
        if len(kept) == num_new_responses:
            break

    if len(kept) < num_new_responses and not allow_short_generation:
        raise ValueError(
            f"Only {len(kept)}/{num_new_responses} generated responses survived filtering. "
            "Use --allow_short_generation or adjust generation settings."
        )
    return kept, dropped


def generate_for_groups(
    groups: list[PromptGroup],
    generate_fn: GenerateFn,
    *,
    batch_size: int,
    num_new_responses: int,
    deduplicate: bool,
    allow_short_generation: bool,
    disable_tqdm: bool = False,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    if batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")

    generated_by_key: dict[str, list[str]] = {}
    stats = {"prompt_groups": len(groups), "generated_responses": 0, "dropped_generated_responses": 0}

    batch_starts = range(0, len(groups), batch_size)
    for start in tqdm(
        batch_starts,
        total=(len(groups) + batch_size - 1) // batch_size,
        desc="Generating responses",
        unit="batch",
        disable=disable_tqdm,
    ):
        batch_groups = groups[start : start + batch_size]
        batch_outputs = generate_fn([group.prompt for group in batch_groups])
        if len(batch_outputs) != len(batch_groups):
            raise ValueError(
                f"Generator returned {len(batch_outputs)} outputs for a batch of {len(batch_groups)} prompts."
            )

        for group, generated in zip(batch_groups, batch_outputs):
            kept, dropped = filter_generated_responses(
                generated,
                blocked_responses=group.original_responses,
                num_new_responses=num_new_responses,
                deduplicate=deduplicate,
                allow_short_generation=allow_short_generation,
            )
            generated_by_key[group.key] = kept
            stats["generated_responses"] += len(kept)
            stats["dropped_generated_responses"] += dropped

    return generated_by_key, stats


def _ranked_prefix_length(row: dict[str, Any], num_responses: int) -> int:
    value = row.get("ranked_prefix_length", num_responses)
    return max(0, min(int(value), num_responses))


def augment_dataset_dict(
    dataset_dict: DatasetDict,
    generate_fn: GenerateFn,
    *,
    splits: list[str] | None,
    augment_splits: list[str] | None = None,
    max_rows: int | None,
    dataset_downsample_ratio: float = 1.0,
    dataset_downsample_seed: int | None = None,
    dataset_downsample_group_key: str = "source_index",
    dataset_downsample_splits: list[str] | None = None,
    disable_tqdm: bool = False,
    batch_size: int,
    num_new_responses: int,
    deduplicate: bool,
    allow_short_generation: bool,
    augmentation_model: str,
    generation_seed: int,
) -> tuple[DatasetDict, dict[str, Any]]:
    split_names = selected_splits(dataset_dict, splits)
    if augment_splits is None:
        augment_splits = ["train"] if "train" in split_names else list(split_names)
    if dataset_downsample_splits is None:
        dataset_downsample_splits = ["train"] if "train" in split_names else []
    augment_split_names = selected_splits(dataset_dict, augment_splits)
    missing_augment_outputs = [split for split in augment_split_names if split not in split_names]
    if missing_augment_outputs:
        raise ValueError(
            f"`augment_splits` must be included in output `splits`; missing {missing_augment_outputs}."
        )

    effective_downsample_seed = generation_seed if dataset_downsample_seed is None else dataset_downsample_seed
    prepared = prepare_output_splits(
        dataset_dict,
        output_splits=split_names,
        max_rows=max_rows,
        downsample_ratio=dataset_downsample_ratio,
        downsample_seed=effective_downsample_seed,
        downsample_group_key=dataset_downsample_group_key,
        downsample_splits=dataset_downsample_splits,
    )
    groups = collect_prompt_groups(prepared, augment_split_names)
    generated_by_key, generation_stats = generate_for_groups(
        groups,
        generate_fn,
        batch_size=batch_size,
        num_new_responses=num_new_responses,
        deduplicate=deduplicate,
        allow_short_generation=allow_short_generation,
        disable_tqdm=disable_tqdm,
    )

    augmented_splits: dict[str, Dataset] = {}
    for split in tqdm(split_names, desc="Augmenting splits", unit="split", disable=disable_tqdm):
        rows = []
        should_augment = split in augment_split_names
        validate_preformatted_split(prepared[split], split)
        for row in tqdm(
            prepared[split],
            desc=f"Writing {split}",
            unit="row",
            leave=False,
            disable=disable_tqdm,
        ):
            row = dict(row)
            original_responses = _stringify_responses(row["responses"])
            original_scores = [float(score) for score in row["scores"]]
            key = split_generation_key(split, row)
            generated_responses = generated_by_key[key] if should_augment else []
            original_ranked_prefix = _ranked_prefix_length(row, len(original_responses))

            row["responses"] = original_responses + generated_responses
            row["scores"] = original_scores + [0.0] * len(generated_responses)
            row["ranked_prefix_length"] = original_ranked_prefix
            row["augmentation_model"] = augmentation_model
            row["num_generated_responses"] = len(generated_responses)
            row["generation_seed"] = int(generation_seed)
            row["generation_prompt_key"] = key
            rows.append(row)
        augmented_splits[split] = Dataset.from_list(rows)

    augmented = DatasetDict(augmented_splits)
    stats: dict[str, Any] = {
        **generation_stats,
        "splits": split_names,
        "augment_splits": augment_split_names,
        "input_rows": {split: len(prepared[split]) for split in split_names},
        "output_rows": {split: len(augmented[split]) for split in split_names},
        "dataset_downsample_ratio": dataset_downsample_ratio,
        "dataset_downsample_seed": effective_downsample_seed,
        "dataset_downsample_group_key": dataset_downsample_group_key,
        "dataset_downsample_splits": dataset_downsample_splits,
        "num_new_responses": num_new_responses,
        "deduplicate": deduplicate,
        "allow_short_generation": allow_short_generation,
        "augmentation_model": augmentation_model,
        "generation_seed": generation_seed,
    }
    return augmented, stats


def _torch_dtype_from_arg(value: str) -> Any:
    if value == "auto":
        return "auto"
    if not hasattr(torch, value):
        raise ValueError(f"Unknown torch dtype {value!r}. Use 'auto' or a torch dtype name like 'bfloat16'.")
    return getattr(torch, value)


class TransformersGenerator:
    def __init__(self, args: argparse.Namespace):
        self.num_new_responses = args.num_new_responses
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.prompt_format = args.prompt_format

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {"torch_dtype": _torch_dtype_from_arg(args.torch_dtype)}
        if args.device_map != "none":
            model_kwargs["device_map"] = args.device_map
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        self.model.eval()

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        if self.prompt_format == "raw":
            return prompts
        if getattr(self.tokenizer, "chat_template", None) is None:
            LOGGER.warning("prompt_format=chat_auto requested, but tokenizer has no chat_template; using raw prompts.")
            return prompts
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    @torch.inference_mode()
    def __call__(self, prompts: list[str]) -> list[list[str]]:
        formatted_prompts = self._format_prompts(prompts)
        inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True)
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.num_new_responses,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_width = inputs["input_ids"].shape[1]
        decoded: list[list[str]] = [[] for _ in prompts]
        for prompt_idx in range(len(prompts)):
            for sample_idx in range(self.num_new_responses):
                output_idx = prompt_idx * self.num_new_responses + sample_idx
                generated_ids = output_ids[output_idx, input_width:]
                decoded[prompt_idx].append(self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        return decoded


def save_augmented_dataset(dataset_dict: DatasetDict, output_dir: str, stats: dict[str, Any], overwrite: bool) -> None:
    output_path = Path(output_dir)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite or choose a new --output_dir.")
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    with (output_path / "augmentation_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    args = parse_args()

    if args.num_new_responses < 1:
        raise ValueError("`num_new_responses` must be at least 1.")
    if args.max_new_tokens < 1:
        raise ValueError("`max_new_tokens` must be at least 1.")
    if args.temperature <= 0 and args.num_new_responses > 1:
        raise ValueError("`temperature` must be > 0 when generating multiple responses per prompt.")

    LOGGER.info("Loading ranking dataset %s", args.dataset_name)
    dataset_dict = load_ranking_dataset(args.dataset_name, args.dataset_config)
    generator = TransformersGenerator(args)
    augmented, stats = augment_dataset_dict(
        dataset_dict,
        generator,
        splits=args.splits,
        augment_splits=args.augment_splits,
        max_rows=args.max_rows,
        dataset_downsample_ratio=args.dataset_downsample_ratio,
        dataset_downsample_seed=args.dataset_downsample_seed,
        dataset_downsample_group_key=args.dataset_downsample_group_key,
        dataset_downsample_splits=args.dataset_downsample_splits,
        disable_tqdm=args.disable_tqdm,
        batch_size=args.batch_size,
        num_new_responses=args.num_new_responses,
        deduplicate=args.deduplicate,
        allow_short_generation=args.allow_short_generation,
        augmentation_model=args.model_name_or_path,
        generation_seed=args.seed,
    )
    stats.update(
        {
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "max_rows": args.max_rows,
            "augment_splits": args.augment_splits,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "prompt_format": args.prompt_format,
            "disable_tqdm": args.disable_tqdm,
        }
    )
    LOGGER.info("Saving augmented dataset to %s", args.output_dir)
    save_augmented_dataset(augmented, args.output_dir, stats, overwrite=args.overwrite)
    LOGGER.info("Augmentation complete: %s", stats)


if __name__ == "__main__":
    main()
