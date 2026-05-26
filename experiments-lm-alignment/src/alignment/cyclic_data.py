from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict


DIMENSIONS = (
    "instruction_following",
    "honesty",
    "truthfulness",
    "helpfulness",
)


@dataclass
class CyclicFilterStats:
    total_rows: int = 0
    eligible_rows: int = 0
    valid_score_rows: int = 0
    cycle_rows: int = 0


@dataclass
class DisagreementFilterStats:
    total_rows: int = 0
    eligible_rows: int = 0
    valid_score_rows: int = 0
    valid_ranking_rows: int = 0
    valid_strict_ranking_rows: int = 0
    tie_limited_rows: int = 0
    distinct_ranking_rows: int = 0
    disagreement_rows: int = 0


@dataclass
class PatternFilterStats:
    total_rows: int = 0
    eligible_rows: int = 0
    valid_score_rows: int = 0
    tie_limited_rows: int = 0
    pattern_match_rows: int = 0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
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
            return "\n".join(str(item.get("content", "")) for item in value if isinstance(item, dict))
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
    return str(value)


def _extract_dimension_score(
    completion: dict[str, Any],
    dimension: str,
    scores_key: str,
    annotations_key: str,
) -> float | None:
    if isinstance(completion.get(scores_key), dict) and dimension in completion[scores_key]:
        value = completion[scores_key][dimension]
        direct = _to_float(value)
        if direct is not None:
            return direct
        if isinstance(value, dict):
            for nested_key in ("Rating", "rating", "score", "Score"):
                nested_score = _to_float(value.get(nested_key))
                if nested_score is not None:
                    return nested_score

    if isinstance(completion.get(annotations_key), dict) and dimension in completion[annotations_key]:
        value = completion[annotations_key][dimension]
        direct = _to_float(value)
        if direct is not None:
            return direct
        if isinstance(value, dict):
            for nested_key in ("Rating", "rating", "score", "Score"):
                nested_score = _to_float(value.get(nested_key))
                if nested_score is not None:
                    return nested_score

    if dimension in completion:
        return _to_float(completion[dimension])

    return None


def _has_strict_order(scores: list[float], order: tuple[int, int, int, int]) -> bool:
    ordered_scores = [scores[idx] for idx in order]
    return all(ordered_scores[i] > ordered_scores[i + 1] for i in range(len(ordered_scores) - 1))


def _rotate_order(order: tuple[int, int, int, int], shift: int) -> tuple[int, int, int, int]:
    shift = shift % len(order)
    return order[shift:] + order[:shift]


def _strict_desc_order(scores: list[float]) -> tuple[int, int, int, int] | None:
    if len(scores) != 4:
        return None
    order = tuple(sorted(range(4), key=lambda idx: scores[idx], reverse=True))
    if not _has_strict_order(scores, order):
        return None
    return order


def _validate_dimensions(dimensions: tuple[str, ...]) -> tuple[str, ...]:
    if len(dimensions) < 2:
        raise ValueError("`dimensions` must contain at least 2 entries for rotated filtering.")
    if len(dimensions) > 4:
        raise ValueError("`dimensions` can contain at most 4 entries.")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("`dimensions` must not contain duplicates.")
    invalid = [dim for dim in dimensions if dim not in DIMENSIONS]
    if invalid:
        raise ValueError(f"Unknown dimensions: {invalid}. Valid values: {list(DIMENSIONS)}")
    return dimensions


def _validate_dimension_pair(dimensions: tuple[str, ...]) -> tuple[str, str]:
    validated = _validate_dimensions(dimensions)
    if len(validated) != 2:
        raise ValueError("`dimension_pair` must contain exactly 2 dimensions.")
    return validated[0], validated[1]


def parse_rank_pattern(pattern: str) -> tuple[int, int, int, int]:
    labels = {"A": 0, "B": 1, "C": 2, "D": 3}
    pattern = pattern.strip().upper()
    if not pattern:
        raise ValueError("Ranking pattern must not be empty.")

    if ">" in pattern:
        parts = [part.strip() for part in pattern.split(">")]
    elif "," in pattern:
        parts = [part.strip() for part in pattern.split(",")]
    else:
        parts = list(pattern)

    if len(parts) != 4:
        raise ValueError("Ranking pattern must contain exactly A, B, C, and D.")
    if any(part not in labels for part in parts):
        raise ValueError("Ranking pattern may only contain labels A, B, C, and D.")
    if len(set(parts)) != 4:
        raise ValueError("Ranking pattern must not contain duplicate labels.")

    return tuple(labels[part] for part in parts)


def format_rank_pattern(order: tuple[int, int, int, int]) -> str:
    index_to_label = ("A", "B", "C", "D")
    if sorted(order) != [0, 1, 2, 3]:
        raise ValueError("Ranking order must be a permutation of 0, 1, 2, and 3.")
    return ">".join(index_to_label[idx] for idx in order)


def _infer_shift(base_order: tuple[int, int, int, int], dim_order: tuple[int, int, int, int]) -> int | None:
    for shift in range(1, len(base_order)):
        if _rotate_order(base_order, shift) == dim_order:
            return shift
    return None


def _find_rotated_cycle_layout(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...],
) -> tuple[tuple[int, int, int, int], dict[str, int]] | None:
    dimensions = _validate_dimensions(dimensions)
    base_dimension = dimensions[0]
    base_scores = scores_by_dimension.get(base_dimension)
    if base_scores is None:
        return None
    base_order = _strict_desc_order(base_scores)
    if base_order is None:
        return None

    shifts: dict[str, int] = {base_dimension: 0}
    used_shifts = {0}

    for dimension in dimensions[1:]:
        if dimension not in scores_by_dimension:
            return None
        scores = scores_by_dimension[dimension]
        dim_order = _strict_desc_order(scores)
        if dim_order is None:
            return None
        shift = _infer_shift(base_order, dim_order)
        if shift is None:
            return None
        if shift in used_shifts:
            return None
        shifts[dimension] = shift
        used_shifts.add(shift)

    return base_order, shifts


def find_strict_rotated_cycle_base_order(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> tuple[int, int, int, int] | None:
    """
    Return the base response order (A, B, C, D) if dimensions form a strict 4-cycle.

        Any permutation is accepted for the base order. The first selected dimension defines
        the base order P=[A, B, C, D]. Every other selected dimension must be a non-zero
        rotation of P, and non-zero shifts must be distinct across dimensions.
    """
    layout = _find_rotated_cycle_layout(scores_by_dimension, dimensions)
    if layout is None:
        return None
    return layout[0]


def has_strict_rotated_cycle(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> bool:
    return _find_rotated_cycle_layout(scores_by_dimension, dimensions) is not None


def _find_strict_ranking_orders(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...],
) -> dict[str, tuple[int, int, int, int]] | None:
    dimensions = _validate_dimensions(dimensions)
    orders: dict[str, tuple[int, int, int, int]] = {}

    for dimension in dimensions:
        scores = scores_by_dimension.get(dimension)
        if scores is None:
            return None
        order = _strict_desc_order(scores)
        if order is None:
            return None
        orders[dimension] = order

    return orders


def has_strict_ranking_disagreement(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...] = DIMENSIONS,
    *,
    min_distinct_rankings: int = 2,
) -> bool:
    orders = _find_strict_ranking_orders(scores_by_dimension, dimensions)
    if orders is None:
        return False
    return len(set(orders.values())) >= min_distinct_rankings


def _weak_desc_order_signature(scores: list[float]) -> tuple[int, int, int, int] | None:
    if len(scores) != 4:
        return None
    unique_scores = sorted(set(scores), reverse=True)
    if len(unique_scores) < 2:
        return None
    score_to_rank = {score: rank for rank, score in enumerate(unique_scores)}
    return tuple(score_to_rank[score] for score in scores)


def _num_tied_pairs(scores: list[float]) -> int:
    counts = {score: scores.count(score) for score in set(scores)}
    return sum(count * (count - 1) // 2 for count in counts.values() if count > 1)


def _passes_tie_limit(scores_by_dimension: dict[str, list[float]], dimensions: tuple[str, ...], limit: int | None) -> bool:
    if limit is None:
        return True
    return all(
        dimension in scores_by_dimension and _num_tied_pairs(scores_by_dimension[dimension]) <= limit
        for dimension in dimensions
    )


def _find_weak_ranking_signatures(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...],
) -> dict[str, tuple[int, int, int, int]] | None:
    dimensions = _validate_dimensions(dimensions)
    signatures: dict[str, tuple[int, int, int, int]] = {}

    for dimension in dimensions:
        scores = scores_by_dimension.get(dimension)
        if scores is None:
            return None
        signature = _weak_desc_order_signature(scores)
        if signature is None:
            return None
        signatures[dimension] = signature

    return signatures


def has_ranking_disagreement(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...] = DIMENSIONS,
    *,
    allow_ties: bool = False,
    max_ties_per_dimension: int | None = None,
    min_distinct_rankings: int = 2,
) -> bool:
    dimensions = _validate_dimensions(dimensions)
    if min_distinct_rankings < 2:
        raise ValueError("`min_distinct_rankings` must be at least 2.")
    if min_distinct_rankings > len(dimensions):
        raise ValueError("`min_distinct_rankings` cannot exceed the number of selected dimensions.")

    if not allow_ties:
        return has_strict_ranking_disagreement(
            scores_by_dimension,
            dimensions,
            min_distinct_rankings=min_distinct_rankings,
        )

    if not _passes_tie_limit(scores_by_dimension, dimensions, max_ties_per_dimension):
        return False

    signatures = _find_weak_ranking_signatures(scores_by_dimension, dimensions)
    if signatures is None:
        return False
    return len(set(signatures.values())) >= min_distinct_rankings


def _score_desc_order(scores: list[float]) -> tuple[int, int, int, int]:
    return tuple(sorted(range(4), key=lambda idx: (-scores[idx], idx)))


def _validate_rank_patterns(
    patterns: tuple[str | tuple[int, int, int, int], ...],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if len(patterns) != 2:
        raise ValueError("`rank_patterns` must contain exactly 2 patterns.")

    parsed_patterns: list[tuple[int, int, int, int]] = []
    for pattern in patterns:
        if isinstance(pattern, str):
            parsed = parse_rank_pattern(pattern)
        else:
            parsed = tuple(pattern)
            if sorted(parsed) != [0, 1, 2, 3]:
                raise ValueError("Each rank pattern must be a permutation of 0, 1, 2, and 3.")
        parsed_patterns.append(parsed)

    return parsed_patterns[0], parsed_patterns[1]


def _matches_resolved_pattern(
    scores_by_dimension: dict[str, list[float]],
    dimension_pair: tuple[str, str],
    rank_patterns: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> bool:
    return all(
        _score_desc_order(scores_by_dimension[dimension]) == pattern
        for dimension, pattern in zip(dimension_pair, rank_patterns)
    )


def _extract_response_text(completion: Any, response_text_key: str) -> str | None:
    if isinstance(completion, str):
        text = completion.strip()
        return text if text else None
    if not isinstance(completion, dict):
        return None

    text_value = completion.get(response_text_key, completion.get("text", completion.get("content")))
    if text_value is None:
        return None
    text = _as_text(text_value).strip()
    return text if text else None


def _iter_four_candidate_groups(num_candidates: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    combos = list(itertools.combinations(range(num_candidates), 4))
    rng.shuffle(combos)
    return combos


def build_cyclic_rows(
    dataset: Dataset,
    *,
    prompt_column: str = "instruction",
    responses_column: str = "completions",
    response_text_key: str = "response",
    scores_key: str = "scores",
    annotations_key: str = "annotations",
    dimensions: tuple[str, ...] = DIMENSIONS,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], CyclicFilterStats]:
    dimensions = _validate_dimensions(dimensions)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    stats = CyclicFilterStats()

    for row_idx, row in enumerate(dataset):
        stats.total_rows += 1

        prompt_raw = row.get(prompt_column)
        prompt = _as_text(prompt_raw).strip()
        if not prompt:
            continue

        completions = row.get(responses_column)
        if not isinstance(completions, list) or len(completions) < 4:
            continue
        stats.eligible_rows += 1

        found = False
        for candidate_indices in _iter_four_candidate_groups(len(completions), rng):
            selected = [completions[idx] for idx in candidate_indices]

            responses: list[str] = []
            scores_by_dimension: dict[str, list[float]] = {dim: [] for dim in dimensions}
            valid = True

            for completion in selected:
                text = _extract_response_text(completion, response_text_key)
                if text is None:
                    valid = False
                    break
                responses.append(text)

                if not isinstance(completion, dict):
                    valid = False
                    break
                for dimension in dimensions:
                    score = _extract_dimension_score(
                        completion,
                        dimension=dimension,
                        scores_key=scores_key,
                        annotations_key=annotations_key,
                    )
                    if score is None:
                        valid = False
                        break
                    scores_by_dimension[dimension].append(score)
                if not valid:
                    break

            if not valid:
                continue

            stats.valid_score_rows += 1
            layout = _find_rotated_cycle_layout(scores_by_dimension, dimensions=dimensions)
            if layout is not None:
                base_order, shifts = layout
                # Emit one listwise row per dimension, keeping a consistent cyclic candidate set.
                for dimension in dimensions:
                    dim_order = _rotate_order(base_order, shifts[dimension])
                    ordered_responses = [responses[idx] for idx in dim_order]
                    ordered_scores = [scores_by_dimension[dimension][idx] for idx in dim_order]
                    rows.append(
                        {
                            "prompt": prompt,
                            "responses": ordered_responses,
                            "scores": ordered_scores,
                            "preference_dimension": dimension,
                            "source_index": int(row_idx),
                        }
                    )
                stats.cycle_rows += 1
                found = True
                break

        if not found:
            continue

    return rows, stats


def build_disagreement_rows(
    dataset: Dataset,
    *,
    prompt_column: str = "instruction",
    responses_column: str = "completions",
    response_text_key: str = "response",
    scores_key: str = "scores",
    annotations_key: str = "annotations",
    dimensions: tuple[str, ...] = DIMENSIONS,
    seed: int = 0,
    max_source_rows: int | None = None,
    max_examples: int | None = None,
    allow_ties: bool = False,
    max_ties_per_dimension: int | None = None,
    min_distinct_rankings: int = 2,
) -> tuple[list[dict[str, Any]], DisagreementFilterStats]:
    dimensions = _validate_dimensions(dimensions)
    if max_ties_per_dimension is not None and max_ties_per_dimension < 0:
        raise ValueError("`max_ties_per_dimension` must be non-negative when provided.")
    if min_distinct_rankings < 2:
        raise ValueError("`min_distinct_rankings` must be at least 2.")
    if min_distinct_rankings > len(dimensions):
        raise ValueError("`min_distinct_rankings` cannot exceed the number of selected dimensions.")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    stats = DisagreementFilterStats()

    for row_idx, row in enumerate(dataset):
        if max_source_rows is not None and stats.total_rows >= max_source_rows:
            break
        if max_examples is not None and stats.disagreement_rows >= max_examples:
            break

        stats.total_rows += 1

        prompt_raw = row.get(prompt_column)
        prompt = _as_text(prompt_raw).strip()
        if not prompt:
            continue

        completions = row.get(responses_column)
        if not isinstance(completions, list) or len(completions) < 4:
            continue
        stats.eligible_rows += 1

        for candidate_indices in _iter_four_candidate_groups(len(completions), rng):
            selected = [completions[idx] for idx in candidate_indices]

            responses: list[str] = []
            scores_by_dimension: dict[str, list[float]] = {dim: [] for dim in dimensions}
            valid = True

            for completion in selected:
                text = _extract_response_text(completion, response_text_key)
                if text is None:
                    valid = False
                    break
                responses.append(text)

                if not isinstance(completion, dict):
                    valid = False
                    break
                for dimension in dimensions:
                    score = _extract_dimension_score(
                        completion,
                        dimension=dimension,
                        scores_key=scores_key,
                        annotations_key=annotations_key,
                    )
                    if score is None:
                        valid = False
                        break
                    scores_by_dimension[dimension].append(score)
                if not valid:
                    break

            if not valid:
                continue

            stats.valid_score_rows += 1
            strict_orders = _find_strict_ranking_orders(scores_by_dimension, dimensions)
            if strict_orders is not None:
                stats.valid_strict_ranking_rows += 1

            if allow_ties:
                ranking_signatures = _find_weak_ranking_signatures(scores_by_dimension, dimensions)
                if ranking_signatures is None:
                    continue
                stats.valid_ranking_rows += 1
                if not _passes_tie_limit(scores_by_dimension, dimensions, max_ties_per_dimension):
                    continue
                orders = {dimension: _score_desc_order(scores_by_dimension[dimension]) for dimension in dimensions}
            else:
                ranking_signatures = strict_orders
                if strict_orders is None:
                    continue
                stats.valid_ranking_rows += 1
                orders = strict_orders

            stats.tie_limited_rows += 1
            if len(set(ranking_signatures.values())) < min_distinct_rankings:
                continue

            stats.distinct_ranking_rows += 1
            for dimension in dimensions:
                dim_order = orders[dimension]
                ordered_responses = [responses[idx] for idx in dim_order]
                ordered_scores = [scores_by_dimension[dimension][idx] for idx in dim_order]
                rows.append(
                    {
                        "prompt": prompt,
                        "responses": ordered_responses,
                        "scores": ordered_scores,
                        "preference_dimension": dimension,
                        "source_index": int(row_idx),
                    }
                )
            stats.disagreement_rows += 1
            break

    return rows, stats


def build_pattern_rows(
    dataset: Dataset,
    *,
    prompt_column: str = "instruction",
    responses_column: str = "completions",
    response_text_key: str = "response",
    scores_key: str = "scores",
    annotations_key: str = "annotations",
    dimension_pair: tuple[str, str] = ("instruction_following", "helpfulness"),
    rank_patterns: tuple[str | tuple[int, int, int, int], str | tuple[int, int, int, int]] = ("ABCD", "BADC"),
    seed: int = 0,
    max_source_rows: int | None = None,
    max_examples: int | None = None,
    max_ties_per_dimension: int | None = 1,
) -> tuple[list[dict[str, Any]], PatternFilterStats]:
    dimensions = _validate_dimension_pair(tuple(dimension_pair))
    patterns = _validate_rank_patterns(tuple(rank_patterns))
    if max_ties_per_dimension is not None and max_ties_per_dimension < 0:
        raise ValueError("`max_ties_per_dimension` must be non-negative when provided.")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    stats = PatternFilterStats()

    for row_idx, row in enumerate(dataset):
        if max_source_rows is not None and stats.total_rows >= max_source_rows:
            break
        if max_examples is not None and stats.pattern_match_rows >= max_examples:
            break

        stats.total_rows += 1

        prompt_raw = row.get(prompt_column)
        prompt = _as_text(prompt_raw).strip()
        if not prompt:
            continue

        completions = row.get(responses_column)
        if not isinstance(completions, list) or len(completions) < 4:
            continue
        stats.eligible_rows += 1

        for candidate_indices in _iter_four_candidate_groups(len(completions), rng):
            selected = [completions[idx] for idx in candidate_indices]

            responses: list[str] = []
            scores_by_dimension: dict[str, list[float]] = {dim: [] for dim in dimensions}
            valid = True

            for completion in selected:
                text = _extract_response_text(completion, response_text_key)
                if text is None:
                    valid = False
                    break
                responses.append(text)

                if not isinstance(completion, dict):
                    valid = False
                    break
                for dimension in dimensions:
                    score = _extract_dimension_score(
                        completion,
                        dimension=dimension,
                        scores_key=scores_key,
                        annotations_key=annotations_key,
                    )
                    if score is None:
                        valid = False
                        break
                    scores_by_dimension[dimension].append(score)
                if not valid:
                    break

            if not valid:
                continue

            stats.valid_score_rows += 1
            if not _passes_tie_limit(scores_by_dimension, dimensions, max_ties_per_dimension):
                continue
            stats.tie_limited_rows += 1

            if not _matches_resolved_pattern(scores_by_dimension, dimensions, patterns):
                continue

            for dimension, pattern in zip(dimensions, patterns):
                ordered_responses = [responses[idx] for idx in pattern]
                ordered_scores = [scores_by_dimension[dimension][idx] for idx in pattern]
                rows.append(
                    {
                        "prompt": prompt,
                        "responses": ordered_responses,
                        "scores": ordered_scores,
                        "preference_dimension": dimension,
                        "source_index": int(row_idx),
                    }
                )
            stats.pattern_match_rows += 1
            break

    return rows, stats


def split_rows_to_dataset_dict(rows: list[dict[str, Any]], seed: int = 0) -> DatasetDict:
    if not rows:
        raise ValueError("No cyclic rows found. Nothing to split.")

    dataset = Dataset.from_list(rows)
    first_split = dataset.train_test_split(test_size=0.2, seed=seed)
    second_split = first_split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )


def split_rows_to_grouped_dataset_dict(rows: list[dict[str, Any]], seed: int = 0) -> DatasetDict:
    if not rows:
        raise ValueError("No rows found. Nothing to split.")

    source_indices = sorted({int(row["source_index"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(source_indices)

    n = len(source_indices)
    train_end = int(n * 0.8)
    validation_end = train_end + int(n * 0.1)
    split_indices = {
        "train": set(source_indices[:train_end]),
        "validation": set(source_indices[train_end:validation_end]),
        "test": set(source_indices[validation_end:]),
    }
    dataset = Dataset.from_list(rows)
    split_positions = {
        split: [idx for idx, row in enumerate(rows) if int(row["source_index"]) in indices]
        for split, indices in split_indices.items()
    }
    return DatasetDict({split: dataset.select(split_positions[split]) for split in ("train", "validation", "test")})
