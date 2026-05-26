# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
from itertools import combinations
from typing import Any

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets

from .configs import ScriptArguments
from .cyclic_data import DIMENSIONS, build_cyclic_rows


logger = logging.getLogger(__name__)


def _is_preformatted_listwise_split(dataset: Dataset) -> bool:
    required = {"prompt", "responses", "scores", "preference_dimension"}
    return required.issubset(set(dataset.column_names))


def _split_seed(seed: int, split_name: str) -> int:
    return seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(split_name))


def _hashable_group_value(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _should_downsample_split(split_name: str, args: ScriptArguments) -> bool:
    ratio = getattr(args, "dataset_downsample_ratio", 1.0)
    if ratio >= 1.0:
        return False
    split_names = getattr(args, "dataset_downsample_splits", None)
    if split_names is None:
        return True
    return "all" in split_names or split_name in split_names


def _downsample_split(dataset: Dataset, args: ScriptArguments, split_name: str) -> Dataset:
    ratio = getattr(args, "dataset_downsample_ratio", 1.0)
    if len(dataset) == 0 or ratio >= 1.0:
        return dataset

    seed = _split_seed(getattr(args, "dataset_downsample_seed", 0), split_name)
    group_key = getattr(args, "dataset_downsample_group_key", "source_index")
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

        rng = random.Random(seed)
        rng.shuffle(group_values)
        selected_groups = set(group_values[:keep_group_count])
        selected_positions = [
            idx
            for idx, value in enumerate(dataset[group_key])
            if _hashable_group_value(value) in selected_groups
        ]
        logger.info(
            "Downsampled split '%s' by group '%s': kept %d/%d groups and %d/%d rows.",
            split_name,
            group_key,
            keep_group_count,
            len(group_values),
            len(selected_positions),
            len(dataset),
        )
        return dataset.select(selected_positions)

    keep_count = max(1, int(len(dataset) * ratio))
    if keep_count >= len(dataset):
        return dataset
    logger.info(
        "Downsampled split '%s' row-wise: kept %d/%d rows.",
        split_name,
        keep_count,
        len(dataset),
    )
    return dataset.shuffle(seed=seed).select(range(keep_count))


def _maybe_downsample_dataset_dict(dataset_dict: DatasetDict, args: ScriptArguments) -> DatasetDict:
    if getattr(args, "dataset_downsample_ratio", 1.0) >= 1.0:
        return dataset_dict
    return DatasetDict(
        {
            split_name: _downsample_split(split_data, args, split_name)
            if _should_downsample_split(split_name, args)
            else split_data
            for split_name, split_data in dataset_dict.items()
        }
    )


def _limit_preformatted_listwise_split(dataset: Dataset, args: ScriptArguments) -> Dataset:
    rows: list[dict[str, Any]] = []
    target_k = args.listwise_num_responses
    generated_k = getattr(args, "listwise_num_generated_responses", 0)
    min_k = args.listwise_min_responses
    allowed_dimensions = set(args.preference_dimensions) if args.preference_dimensions is not None else None

    for row in dataset:
        row_dimension = row.get("preference_dimension")
        if allowed_dimensions is not None and row_dimension not in allowed_dimensions:
            continue

        responses = row.get("responses")
        scores = row.get("scores")
        if not isinstance(responses, list) or not isinstance(scores, list):
            continue

        n = min(len(responses), len(scores))
        if n < min_k:
            continue

        effective_k = min(target_k, n)
        if effective_k < min_k:
            continue

        trimmed_responses = responses[:n]
        trimmed_scores = [float(x) for x in scores[:n]]

        if generated_k > 0:
            ranked_prefix_length = row.get("ranked_prefix_length", n)
            ranked_prefix_length = max(0, min(int(ranked_prefix_length), n))
            original_indices = tuple(range(ranked_prefix_length))
            generated_indices = tuple(range(ranked_prefix_length, n))

            if len(original_indices) < target_k:
                continue

            if generated_indices:
                if len(generated_indices) < generated_k:
                    continue
                subset_indices_iter = (
                    original_subset + generated_subset
                    for original_subset in combinations(original_indices, target_k)
                    for generated_subset in combinations(generated_indices, generated_k)
                )
            else:
                subset_indices_iter = combinations(original_indices, target_k)

            for idxs in subset_indices_iter:
                new_row = dict(row)
                new_row["responses"] = [trimmed_responses[i] for i in idxs]
                new_row["scores"] = [trimmed_scores[i] for i in idxs]
                new_row["ranked_prefix_length"] = target_k
                rows.append(new_row)
            continue

        if n <= effective_k:
            new_row = dict(row)
            new_row["responses"] = trimmed_responses
            new_row["scores"] = trimmed_scores
            if "ranked_prefix_length" in new_row:
                new_row["ranked_prefix_length"] = int(new_row["ranked_prefix_length"])
            rows.append(new_row)
            continue

        ranked_prefix_length = row.get("ranked_prefix_length")
        if ranked_prefix_length is not None:
            ranked_prefix_length = max(0, min(int(ranked_prefix_length), n))
            if ranked_prefix_length > effective_k:
                continue
            prefix_indices = tuple(range(ranked_prefix_length))
            tail_needed = effective_k - ranked_prefix_length
            subset_indices_iter = (
                prefix_indices + tuple(tail_indices)
                for tail_indices in combinations(range(ranked_prefix_length, n), tail_needed)
            )
        else:
            subset_indices_iter = combinations(range(n), effective_k)

        for idxs in subset_indices_iter:
            new_row = dict(row)
            new_row["responses"] = [trimmed_responses[i] for i in idxs]
            new_row["scores"] = [trimmed_scores[i] for i in idxs]
            if ranked_prefix_length is not None:
                new_row["ranked_prefix_length"] = ranked_prefix_length
            rows.append(new_row)

    if not rows:
        raise ValueError(
            "No listwise examples remain after applying listwise_num_responses to preformatted listwise data. "
            "Check listwise_num_responses/listwise_min_responses settings and dataset columns."
        )
    return Dataset.from_list(rows)


def _pairwise_from_listwise_split(dataset: Dataset, args: ScriptArguments) -> Dataset:
    rows: list[dict[str, Any]] = []
    allowed_dimensions = set(args.preference_dimensions) if args.preference_dimensions is not None else None
    strategy = args.pairwise_from_listwise_strategy

    for row in dataset:
        row_dimension = row.get("preference_dimension")
        if allowed_dimensions is not None and row_dimension not in allowed_dimensions:
            continue

        responses = row.get("responses")
        scores = row.get("scores")
        if not isinstance(responses, list) or not isinstance(scores, list):
            continue

        n = min(len(responses), len(scores))
        if n < 2:
            continue

        trimmed_responses = [str(response) for response in responses[:n]]
        trimmed_scores = [float(score) for score in scores[:n]]
        ranked_prefix_length = row.get("ranked_prefix_length")
        if ranked_prefix_length is not None:
            ranked_prefix_length = max(0, min(int(ranked_prefix_length), n))
            pair_indices = [
                (chosen_idx, rejected_idx)
                for chosen_idx in range(ranked_prefix_length)
                for rejected_idx in range(chosen_idx + 1, n)
            ]
        else:
            pair_indices = [(0, n - 1)] if strategy == "extreme" else list(combinations(range(n), 2))

        for chosen_idx, rejected_idx in pair_indices:
            pairwise_row = {
                "prompt": row["prompt"],
                "chosen": trimmed_responses[chosen_idx],
                "rejected": trimmed_responses[rejected_idx],
                "chosen_score": trimmed_scores[chosen_idx],
                "rejected_score": trimmed_scores[rejected_idx],
                "preference_dimension": row_dimension,
            }
            for metadata_key in (
                "source_index",
                "source_dataset",
                "persona_id",
                "persona_text",
                "instruction",
                "ranked_prefix_length",
            ):
                if metadata_key in row:
                    pairwise_row[metadata_key] = row[metadata_key]
            rows.append(pairwise_row)

    if not rows:
        raise ValueError(
            "No pairwise examples could be built from listwise rankings. "
            "Check pairwise_from_listwise_strategy, preference_dimensions, and dataset columns."
        )
    return Dataset.from_list(rows)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "content" in value[0]:
            return "\n".join(item.get("content", "") for item in value if isinstance(item, dict))
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
    return str(value)


def _extract_dimension_score(
    item: dict[str, Any],
    row: dict[str, Any],
    idx: int,
    dimension: str,
    scores_key: str, # not used in UltraFeedback
    annotations_key: str,
) -> float | None:
    # Most common format: completion[scores_key][dimension]
    if isinstance(item.get(scores_key), dict) and dimension in item[scores_key]:
        direct_score = _to_float(item[scores_key][dimension])
        if direct_score is not None:
            return direct_score

        # Handle nested annotation format: scores[dimension]['Rating']
        nested = item[scores_key][dimension]
        if isinstance(nested, dict):
            for key in ("Rating", "rating", "score", "Score"):
                score = _to_float(nested.get(key))
                if score is not None:
                    return score

    # Fallback format: completion[annotations_key][dimension]
    if isinstance(item.get(annotations_key), dict) and dimension in item[annotations_key]:
        direct_score = _to_float(item[annotations_key][dimension])
        if direct_score is not None:
            return direct_score

        nested = item[annotations_key][dimension]
        if isinstance(nested, dict):
            for key in ("Rating", "rating", "score", "Score"):
                score = _to_float(nested.get(key))
                if score is not None:
                    return score

    # Fallback format: completion[dimension]
    if dimension in item:
        score = _to_float(item[dimension])
        if score is not None:
            return score
    
    # Fallback format: row-level score list aligned with response index
    row_key = f"{dimension}_scores"
    if isinstance(row.get(row_key), list) and idx < len(row[row_key]):
        return _to_float(row[row_key][idx])
    return None


def _extract_candidates(row: dict[str, Any], args: ScriptArguments, dimension: str) -> list[tuple[str, float]]:
    raw_responses = row.get(args.listwise_responses_column)
    if raw_responses is None:
        # Allow common fallback names.
        raw_responses = row.get("responses", row.get("completions"))

    if not isinstance(raw_responses, list):
        return []

    pairs: list[tuple[str, float]] = [] # containing (response_text, score) pairs
    for idx, item in enumerate(raw_responses):
        text: str | None = None
        score: float | None = None

        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text_val = item.get(args.listwise_response_text_key, item.get("text", item.get("content")))
            if text_val is not None:
                text = _as_text(text_val)
            score = _extract_dimension_score(
                item=item,
                row=row,
                idx=idx,
                dimension=dimension,
                scores_key=args.listwise_scores_key,
                annotations_key=args.listwise_annotations_key,
            )

        if text is None or score is None:
            continue
        text = text.strip()
        if not text:
            continue
        pairs.append((text, score))

    if not pairs:
        return []
    return pairs


def _to_listwise_dataset(dataset: Dataset, args: ScriptArguments) -> Dataset:
    if args.preference_dimensions is None:
        raise ValueError("`preference_dimensions` must be set for listwise dataset conversion")

    if args.listwise_use_cyclic_filter:
        dimensions = tuple(args.preference_dimensions) if args.preference_dimensions else DIMENSIONS
        seed = args.listwise_seed
        rows, stats = build_cyclic_rows(
            dataset,
            prompt_column=args.listwise_prompt_column,
            responses_column=args.listwise_responses_column,
            response_text_key=args.listwise_response_text_key,
            scores_key=args.listwise_scores_key,
            annotations_key=args.listwise_annotations_key,
            dimensions=dimensions,
            seed=seed,
        )
        logger.info(
            "Cyclic listwise filtering kept %d/%d eligible rows (%d valid candidate sets)",
            stats.cycle_rows,
            stats.eligible_rows,
            stats.valid_score_rows,
        )
        if not rows:
            raise ValueError(
                "No cyclic listwise examples could be built from the dataset. "
                "Try another split, dimensions, or annotation key settings."
            )
        return Dataset.from_list(rows)

    rows = []
    dimensions = tuple(args.preference_dimensions)
    for row in dataset:
        raw_prompt = row.get(args.listwise_prompt_column)
        prompt = _as_text(raw_prompt).strip()
        if not prompt:
            continue

        for dimension in dimensions:
            candidates = _extract_candidates(row, args, dimension=dimension)
            if len(candidates) < args.listwise_min_responses:
                continue

            ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
            if len(ranked) < args.listwise_min_responses:
                continue

            effective_k = min(args.listwise_num_responses, len(ranked))
            if effective_k < args.listwise_min_responses:
                continue

            if len(ranked) == effective_k:
                subset_rankings = [ranked]
            else:
                subset_rankings = [list(subset) for subset in combinations(ranked, effective_k)]

            for subset in subset_rankings:
                subset = sorted(subset, key=lambda x: x[1], reverse=True)
                responses = [x[0] for x in subset]
                scores = [float(x[1]) for x in subset]
                rows.append(
                    {
                        "prompt": prompt,
                        "responses": responses,
                        "scores": scores,
                        "preference_dimension": dimension,
                    }
                )

    if not rows:
        raise ValueError(
            "No listwise examples could be built from the dataset. "
            "Check listwise column/key settings and preference_dimensions."
        )
    return Dataset.from_list(rows)


def _maybe_convert_to_listwise(dataset_dict: DatasetDict, args: ScriptArguments) -> DatasetDict:
    if args.dataset_format != "listwise":
        return dataset_dict

    converted = {}
    logged_dimensions = args.preference_dimensions if args.preference_dimensions is not None else []
    for split_name, split_data in dataset_dict.items():
        if _is_preformatted_listwise_split(split_data):
            logger.info(
                "Split '%s' already has listwise columns; applying listwise_num_responses=%d reduction if needed.",
                split_name,
                args.listwise_num_responses,
            )
            converted[split_name] = _limit_preformatted_listwise_split(split_data, args)
            logger.info(
                "Preformatted split '%s' produced %d listwise examples after reduction.",
                split_name,
                len(converted[split_name]),
            )
            continue

        logger.info(
            "Converting split '%s' to listwise format for dimensions '%s'",
            split_name,
            logged_dimensions,
        )
        converted[split_name] = _to_listwise_dataset(split_data, args)
        logger.info("Built %d listwise examples for split '%s'", len(converted[split_name]), split_name)
    return DatasetDict(converted)


def _maybe_convert_to_pairwise(dataset_dict: DatasetDict, args: ScriptArguments) -> DatasetDict:
    if args.dataset_format != "pairwise":
        return dataset_dict

    converted = {}
    allowed_dimensions = set(args.preference_dimensions) if args.preference_dimensions is not None else None
    for split_name, split_data in dataset_dict.items():
        if not _is_preformatted_listwise_split(split_data):
            if allowed_dimensions is not None and "preference_dimension" in split_data.column_names:
                converted[split_name] = split_data.filter(
                    lambda row: row.get("preference_dimension") in allowed_dimensions,
                    desc=f"Filtering {split_name} by preference dimension",
                )
                logger.info(
                    "Filtered pairwise split '%s' to %d examples for dimensions '%s'.",
                    split_name,
                    len(converted[split_name]),
                    args.preference_dimensions,
                )
            else:
                converted[split_name] = split_data
            continue

        logger.info(
            "Split '%s' has listwise ranking columns but dataset_format='pairwise'; converting with strategy='%s'.",
            split_name,
            args.pairwise_from_listwise_strategy,
        )
        converted[split_name] = _pairwise_from_listwise_split(split_data, args)
        logger.info("Built %d pairwise examples for split '%s'", len(converted[split_name]), split_name)
    return DatasetDict(converted)


def _load_named_dataset(args: ScriptArguments) -> DatasetDict:
    logger.info(f"Loading dataset: {args.dataset_name}")
    if os.path.isdir(args.dataset_name):
        logger.info("Detected local dataset directory, trying datasets.load_from_disk")
        dataset = datasets.load_from_disk(args.dataset_name)
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
    else:
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
    return dataset


def get_ranking_dataset(args: ScriptArguments) -> DatasetDict | None:
    """Load ranking-shaped splits for offline ranking evaluation, if available."""
    if args.dataset_name is None or args.dataset_mixture is not None:
        return None

    dataset = _load_named_dataset(args)
    dataset = _maybe_downsample_dataset_dict(dataset, args)
    if all(_is_preformatted_listwise_split(split_data) for split_data in dataset.values()):
        return DatasetDict(
            {
                split_name: _limit_preformatted_listwise_split(split_data, args)
                for split_name, split_data in dataset.items()
            }
        )
    if args.dataset_format == "listwise":
        return _maybe_convert_to_listwise(dataset, args)
    return None


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        dataset = _load_named_dataset(args)
        dataset = _maybe_downsample_dataset_dict(dataset, args)
        dataset = _maybe_convert_to_listwise(dataset, args)
        return _maybe_convert_to_pairwise(dataset, args)
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            if os.path.isdir(dataset_config.id):
                logger.info("Detected local mixture dataset directory, trying datasets.load_from_disk")
                local_ds = datasets.load_from_disk(dataset_config.id)
                if isinstance(local_ds, DatasetDict):
                    if dataset_config.split not in local_ds:
                        raise ValueError(
                            f"Requested split '{dataset_config.split}' not found in local dataset '{dataset_config.id}'. "
                            f"Available splits: {list(local_ds.keys())}"
                        )
                    ds = local_ds[dataset_config.split]
                elif isinstance(local_ds, Dataset):
                    if dataset_config.split != "train":
                        raise ValueError(
                            f"Local dataset '{dataset_config.id}' is a single split dataset. "
                            f"Requested split '{dataset_config.split}' is not available; use split='train'."
                        )
                    ds = local_ds
                else:
                    raise ValueError(
                        f"Unsupported object loaded from local dataset '{dataset_config.id}': {type(local_ds)}"
                    )
            else:
                ds = datasets.load_dataset(
                    dataset_config.id,
                    dataset_config.config,
                    split=dataset_config.split,
                )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                combined_dataset = _maybe_downsample_dataset_dict(combined_dataset, args)
                combined_dataset = _maybe_convert_to_listwise(combined_dataset, args)
                return _maybe_convert_to_pairwise(combined_dataset, args)
            else:
                combined_dataset = _maybe_downsample_dataset_dict(DatasetDict({"train": combined_dataset}), args)
                combined_dataset = _maybe_convert_to_listwise(combined_dataset, args)
                return _maybe_convert_to_pairwise(combined_dataset, args)
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
