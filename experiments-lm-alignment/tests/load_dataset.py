# %%

from datasets import load_from_disk
dimensions = ["instruction_following", "helpfulness", "truthfulness", "honesty"]

# for i in range(4):
#     for j in range(i + 1, 4):
path = f'../data/ultrafeedback_disagreement_train_augmented_Qwen3_0p6B_ds0p25_k8_s42'
ds = load_from_disk(path)
# %%

import logging
import os
from itertools import combinations
from typing import Any

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets


logger = logging.getLogger(__name__)


class args:
    dataset_name = "./data/cyclic_ultrafeedback_m2_instruction_following_helpfulness"
    listwise_responses_column = "completions"
    listwise_response_text_key = "response"
    listwise_scores_key = "scores"
    listwise_annotations_key = "annotations"
    listwise_num_responses = 2
    listwise_min_responses = 2
    preference_dimensions = ["instruction_following"]
    listwise_use_cyclic_filter = False
    dataset_format = "listwise"
    

def _is_preformatted_listwise_split(dataset: Dataset) -> bool:
    required = {"prompt", "responses", "scores", "preference_dimension"}
    return required.issubset(set(dataset.column_names))


def _limit_preformatted_listwise_split(dataset: Dataset, args) -> Dataset:
    rows: list[dict[str, Any]] = []
    target_k = args.listwise_num_responses
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

        if n <= effective_k:
            new_row = dict(row)
            new_row["responses"] = trimmed_responses
            new_row["scores"] = trimmed_scores
            rows.append(new_row)
            continue

        for idxs in combinations(range(n), effective_k):
            new_row = dict(row)
            new_row["responses"] = [trimmed_responses[i] for i in idxs]
            new_row["scores"] = [trimmed_scores[i] for i in idxs]
            rows.append(new_row)

    if not rows:
        raise ValueError(
            "No listwise examples remain after applying listwise_num_responses to preformatted listwise data. "
            "Check listwise_num_responses/listwise_min_responses settings and dataset columns."
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


def _extract_candidates(row: dict[str, Any], args, dimension: str) -> list[tuple[str, float]]:
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


def _to_listwise_dataset(dataset: Dataset, args) -> Dataset:
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


def _maybe_convert_to_listwise(dataset_dict: DatasetDict, args) -> DatasetDict:
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

dataset = _maybe_convert_to_listwise(ds, args)
# %%
