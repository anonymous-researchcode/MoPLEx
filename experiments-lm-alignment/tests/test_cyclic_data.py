# coding=utf-8

import pytest
from datasets import Dataset

from alignment.cyclic_data import (
    build_cyclic_rows,
    build_disagreement_rows,
    build_pattern_rows,
    format_rank_pattern,
    has_ranking_disagreement,
    has_strict_ranking_disagreement,
    has_strict_rotated_cycle,
    parse_rank_pattern,
    split_rows_to_dataset_dict,
    split_rows_to_grouped_dataset_dict,
)


def _make_cyclic_completion(text: str, instr: float, honesty: float, truth: float, helpf: float):
    return {
        "response": text,
        "annotations": {
            "instruction_following": {"Rating": instr},
            "honesty": {"Rating": honesty},
            "truthfulness": {"Rating": truth},
            "helpfulness": {"Rating": helpf},
        },
    }


def test_has_strict_rotated_cycle_accepts_expected_pattern():
    # A, B, C, D aligned at indices 0..3.
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [1.0, 4.0, 3.0, 2.0],
        "truthfulness": [2.0, 1.0, 4.0, 3.0],
        "helpfulness": [3.0, 2.0, 1.0, 4.0],
    }
    assert has_strict_rotated_cycle(scores)


def test_has_strict_rotated_cycle_rejects_ties():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [1.0, 4.0, 3.0, 2.0],
        "truthfulness": [2.0, 1.0, 4.0, 3.0],
        "helpfulness": [3.0, 2.0, 2.0, 4.0],
    }
    assert not has_strict_rotated_cycle(scores)


def test_build_cyclic_rows_keeps_only_matching_example():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-good",
                "completions": [
                    _make_cyclic_completion("A", 4, 1, 2, 3),
                    _make_cyclic_completion("B", 3, 4, 1, 2),
                    _make_cyclic_completion("C", 2, 3, 4, 1),
                    _make_cyclic_completion("D", 1, 2, 3, 4),
                ],
            },
            {
                "instruction": "p-bad",
                "completions": [
                    _make_cyclic_completion("A", 4, 4, 4, 4),
                    _make_cyclic_completion("B", 3, 3, 3, 3),
                    _make_cyclic_completion("C", 2, 2, 2, 2),
                    _make_cyclic_completion("D", 1, 1, 1, 1),
                ],
            },
        ]
    )

    rows, stats = build_cyclic_rows(dataset, seed=123)

    assert stats.total_rows == 2
    assert stats.eligible_rows == 2
    assert stats.cycle_rows == 1
    assert len(rows) == 4
    assert all(row["prompt"] == "p-good" for row in rows)
    assert all(len(row["responses"]) == 4 for row in rows)
    assert all(len(row["scores"]) == 4 for row in rows)
    assert all(row["scores"] == sorted(row["scores"], reverse=True) for row in rows)
    assert {row["preference_dimension"] for row in rows} == {
        "instruction_following",
        "honesty",
        "truthfulness",
        "helpfulness",
    }


def test_has_strict_rotated_cycle_accepts_any_permutation_base_order():
    # Base order is [2, 0, 3, 1] instead of fixed [0, 1, 2, 3].
    scores = {
        "instruction_following": [3.0, 1.0, 4.0, 2.0],
        "honesty": [4.0, 2.0, 1.0, 3.0],
        "truthfulness": [1.0, 3.0, 2.0, 4.0],
        "helpfulness": [2.0, 4.0, 3.0, 1.0],
    }
    assert has_strict_rotated_cycle(scores)


def test_has_strict_rotated_cycle_subset_dimensions_m2():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        # rotate by 1 relative to instruction_following
        "helpfulness": [1.0, 4.0, 3.0, 2.0],
    }
    assert has_strict_rotated_cycle(scores, dimensions=("instruction_following", "helpfulness"))


def test_has_strict_rotated_cycle_subset_dimensions_m2_allows_shift_2():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        # rotate by 2 relative to instruction_following
        "helpfulness": [2.0, 1.0, 4.0, 3.0],
    }
    assert has_strict_rotated_cycle(scores, dimensions=("instruction_following", "helpfulness"))


def test_has_strict_rotated_cycle_rejects_duplicate_nonzero_shifts():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        # both rotate by 1 from base -> should be rejected because shifts must differ
        "honesty": [1.0, 4.0, 3.0, 2.0],
        "helpfulness": [1.0, 4.0, 3.0, 2.0],
    }
    assert not has_strict_rotated_cycle(scores, dimensions=("instruction_following", "honesty", "helpfulness"))


def test_build_cyclic_rows_subset_dimensions_emits_only_requested_rows():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-subset",
                "completions": [
                    _make_cyclic_completion("A", 4, 1, 2, 1),
                    _make_cyclic_completion("B", 3, 4, 1, 4),
                    _make_cyclic_completion("C", 2, 3, 4, 3),
                    _make_cyclic_completion("D", 1, 2, 3, 2),
                ],
            }
        ]
    )

    rows, stats = build_cyclic_rows(dataset, dimensions=("instruction_following", "helpfulness"), seed=0)

    assert stats.cycle_rows == 1
    assert len(rows) == 2
    assert {row["preference_dimension"] for row in rows} == {"instruction_following", "helpfulness"}


def test_build_cyclic_rows_finds_combination_when_more_than_four_candidates():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p1",
                "completions": [
                    _make_cyclic_completion("X", 0, 0, 0, 0),
                    _make_cyclic_completion("Y", 0, 0, 0, 0),
                    _make_cyclic_completion("A", 4, 1, 2, 3),
                    _make_cyclic_completion("B", 3, 4, 1, 2),
                    _make_cyclic_completion("C", 2, 3, 4, 1),
                    _make_cyclic_completion("D", 1, 2, 3, 4),
                ],
            }
        ]
    )

    rows, stats = build_cyclic_rows(dataset, seed=0)

    assert stats.total_rows == 1
    assert stats.eligible_rows == 1
    assert stats.cycle_rows == 1
    assert len(rows) == 4
    assert {row["preference_dimension"] for row in rows} == {
        "instruction_following",
        "honesty",
        "truthfulness",
        "helpfulness",
    }


def test_has_strict_ranking_disagreement_accepts_any_different_strict_order():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [4.0, 2.0, 3.0, 1.0],
        "truthfulness": [4.0, 3.0, 2.0, 1.0],
        "helpfulness": [4.0, 3.0, 2.0, 1.0],
    }

    assert has_strict_ranking_disagreement(scores)


def test_has_strict_ranking_disagreement_rejects_identical_rankings():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [8.0, 7.0, 6.0, 5.0],
        "truthfulness": [0.4, 0.3, 0.2, 0.1],
        "helpfulness": [40.0, 30.0, 20.0, 10.0],
    }

    assert not has_strict_ranking_disagreement(scores)


def test_has_strict_ranking_disagreement_respects_min_distinct_rankings():
    two_distinct = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [4.0, 2.0, 3.0, 1.0],
        "truthfulness": [4.0, 3.0, 2.0, 1.0],
        "helpfulness": [4.0, 3.0, 2.0, 1.0],
    }
    three_distinct = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [4.0, 2.0, 3.0, 1.0],
        "truthfulness": [3.0, 4.0, 2.0, 1.0],
        "helpfulness": [4.0, 3.0, 2.0, 1.0],
    }

    assert has_strict_ranking_disagreement(two_distinct)
    assert not has_strict_ranking_disagreement(two_distinct, min_distinct_rankings=3)
    assert has_strict_ranking_disagreement(three_distinct, min_distinct_rankings=3)


def test_has_strict_ranking_disagreement_rejects_ties():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [4.0, 3.0, 3.0, 1.0],
        "truthfulness": [1.0, 4.0, 3.0, 2.0],
        "helpfulness": [2.0, 1.0, 4.0, 3.0],
    }

    assert not has_strict_ranking_disagreement(scores)


def test_has_ranking_disagreement_allow_ties_compares_weak_orders():
    scores = {
        "instruction_following": [4.0, 3.0, 3.0, 1.0],
        "honesty": [4.0, 4.0, 2.0, 1.0],
        "truthfulness": [4.0, 3.0, 3.0, 1.0],
        "helpfulness": [4.0, 3.0, 3.0, 1.0],
    }

    assert not has_ranking_disagreement(scores)
    assert has_ranking_disagreement(scores, allow_ties=True)


def test_has_ranking_disagreement_allow_ties_rejects_identical_weak_orders():
    scores = {
        "instruction_following": [4.0, 3.0, 3.0, 1.0],
        "honesty": [8.0, 7.0, 7.0, 5.0],
        "truthfulness": [0.4, 0.3, 0.3, 0.1],
        "helpfulness": [40.0, 30.0, 30.0, 10.0],
    }

    assert not has_ranking_disagreement(scores, allow_ties=True)


def test_has_ranking_disagreement_allow_ties_rejects_all_equal_dimension():
    scores = {
        "instruction_following": [4.0, 3.0, 3.0, 1.0],
        "honesty": [2.0, 2.0, 2.0, 2.0],
        "truthfulness": [4.0, 3.0, 3.0, 1.0],
        "helpfulness": [4.0, 3.0, 3.0, 1.0],
    }

    assert not has_ranking_disagreement(scores, allow_ties=True)


def test_has_ranking_disagreement_allow_ties_respects_max_ties_per_dimension():
    scores = {
        "instruction_following": [4.0, 3.0, 3.0, 1.0],
        "honesty": [4.0, 4.0, 2.0, 1.0],
        "truthfulness": [4.0, 3.0, 3.0, 1.0],
        "helpfulness": [4.0, 3.0, 3.0, 1.0],
    }
    too_many_ties = {
        "instruction_following": [4.0, 4.0, 2.0, 2.0],
        "honesty": [4.0, 3.0, 3.0, 1.0],
        "truthfulness": [4.0, 3.0, 3.0, 1.0],
        "helpfulness": [4.0, 3.0, 3.0, 1.0],
    }

    assert has_ranking_disagreement(scores, allow_ties=True, max_ties_per_dimension=1)
    assert not has_ranking_disagreement(too_many_ties, allow_ties=True, max_ties_per_dimension=1)


def test_has_ranking_disagreement_allow_ties_respects_min_distinct_rankings():
    scores = {
        "instruction_following": [4.0, 3.0, 3.0, 1.0],
        "honesty": [4.0, 4.0, 2.0, 1.0],
        "truthfulness": [3.0, 4.0, 3.0, 1.0],
        "helpfulness": [4.0, 3.0, 3.0, 1.0],
    }

    assert has_ranking_disagreement(scores, allow_ties=True, min_distinct_rankings=3)
    assert not has_ranking_disagreement(scores, allow_ties=True, min_distinct_rankings=4)


def test_build_disagreement_rows_emits_one_row_per_dimension():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-disagree",
                "completions": [
                    _make_cyclic_completion("A", 4, 4, 1, 4),
                    _make_cyclic_completion("B", 3, 2, 4, 3),
                    _make_cyclic_completion("C", 2, 3, 3, 2),
                    _make_cyclic_completion("D", 1, 1, 2, 1),
                ],
            },
            {
                "instruction": "p-identical",
                "completions": [
                    _make_cyclic_completion("A", 4, 8, 0.4, 40),
                    _make_cyclic_completion("B", 3, 7, 0.3, 30),
                    _make_cyclic_completion("C", 2, 6, 0.2, 20),
                    _make_cyclic_completion("D", 1, 5, 0.1, 10),
                ],
            },
        ]
    )

    rows, stats = build_disagreement_rows(dataset, seed=123)

    assert stats.total_rows == 2
    assert stats.eligible_rows == 2
    assert stats.valid_strict_ranking_rows == 2
    assert stats.disagreement_rows == 1
    assert len(rows) == 4
    assert all(row["prompt"] == "p-disagree" for row in rows)
    assert all(row["scores"] == sorted(row["scores"], reverse=True) for row in rows)
    assert {row["preference_dimension"] for row in rows} == {
        "instruction_following",
        "honesty",
        "truthfulness",
        "helpfulness",
    }


def test_build_disagreement_rows_rejects_tied_dimension():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-tie",
                "completions": [
                    _make_cyclic_completion("A", 4, 1, 1, 4),
                    _make_cyclic_completion("B", 3, 4, 4, 3),
                    _make_cyclic_completion("C", 2, 3, 3, 2),
                    _make_cyclic_completion("D", 1, 2, 3, 1),
                ],
            }
        ]
    )

    rows, stats = build_disagreement_rows(dataset, seed=0)

    assert stats.valid_score_rows == 1
    assert stats.valid_strict_ranking_rows == 0
    assert stats.disagreement_rows == 0
    assert rows == []


def test_build_disagreement_rows_allow_ties_keeps_weak_disagreement():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-weak",
                "completions": [
                    _make_cyclic_completion("A", 4, 4, 4, 4),
                    _make_cyclic_completion("B", 3, 4, 3, 3),
                    _make_cyclic_completion("C", 3, 2, 3, 3),
                    _make_cyclic_completion("D", 1, 1, 1, 1),
                ],
            }
        ]
    )

    strict_rows, strict_stats = build_disagreement_rows(dataset, seed=0)
    weak_rows, weak_stats = build_disagreement_rows(dataset, seed=0, allow_ties=True)

    assert strict_rows == []
    assert strict_stats.valid_ranking_rows == 0
    assert strict_stats.valid_strict_ranking_rows == 0
    assert weak_stats.valid_ranking_rows == 1
    assert weak_stats.valid_strict_ranking_rows == 0
    assert weak_stats.disagreement_rows == 1
    assert len(weak_rows) == 4
    assert all(row["scores"] == sorted(row["scores"], reverse=True) for row in weak_rows)


def test_build_disagreement_rows_allow_ties_respects_max_ties_per_dimension():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-two-ties",
                "completions": [
                    _make_cyclic_completion("A", 4, 4, 4, 4),
                    _make_cyclic_completion("B", 4, 3, 3, 3),
                    _make_cyclic_completion("C", 2, 3, 3, 3),
                    _make_cyclic_completion("D", 2, 1, 1, 1),
                ],
            }
        ]
    )

    uncapped_rows, uncapped_stats = build_disagreement_rows(dataset, seed=0, allow_ties=True)
    capped_rows, capped_stats = build_disagreement_rows(
        dataset,
        seed=0,
        allow_ties=True,
        max_ties_per_dimension=1,
    )

    assert uncapped_stats.valid_ranking_rows == 1
    assert uncapped_stats.tie_limited_rows == 1
    assert uncapped_stats.disagreement_rows == 1
    assert len(uncapped_rows) == 4
    assert capped_stats.valid_ranking_rows == 1
    assert capped_stats.tie_limited_rows == 0
    assert capped_stats.disagreement_rows == 0
    assert capped_rows == []


def test_build_disagreement_rows_respects_min_distinct_rankings():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-two-distinct",
                "completions": [
                    _make_cyclic_completion("A", 4, 4, 4, 4),
                    _make_cyclic_completion("B", 3, 2, 3, 3),
                    _make_cyclic_completion("C", 2, 3, 2, 2),
                    _make_cyclic_completion("D", 1, 1, 1, 1),
                ],
            },
            {
                "instruction": "p-three-distinct",
                "completions": [
                    _make_cyclic_completion("A", 4, 4, 3, 4),
                    _make_cyclic_completion("B", 3, 2, 4, 3),
                    _make_cyclic_completion("C", 2, 3, 2, 2),
                    _make_cyclic_completion("D", 1, 1, 1, 1),
                ],
            },
        ]
    )

    rows, stats = build_disagreement_rows(dataset, seed=0, min_distinct_rankings=3)

    assert stats.valid_ranking_rows == 2
    assert stats.tie_limited_rows == 2
    assert stats.distinct_ranking_rows == 1
    assert stats.disagreement_rows == 1
    assert len(rows) == 4
    assert all(row["prompt"] == "p-three-distinct" for row in rows)


def test_build_disagreement_rows_is_deterministic_for_same_seed():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-many",
                "completions": [
                    _make_cyclic_completion("X", 0, 0, 0, 0),
                    _make_cyclic_completion("Y", 0, 0, 0, 0),
                    _make_cyclic_completion("A", 4, 4, 1, 4),
                    _make_cyclic_completion("B", 3, 2, 4, 3),
                    _make_cyclic_completion("C", 2, 3, 3, 2),
                    _make_cyclic_completion("D", 1, 1, 2, 1),
                ],
            }
        ]
    )

    rows_1, stats_1 = build_disagreement_rows(dataset, seed=11)
    rows_2, stats_2 = build_disagreement_rows(dataset, seed=11)

    assert rows_1 == rows_2
    assert stats_1 == stats_2


def test_parse_rank_pattern_accepts_supported_forms():
    assert parse_rank_pattern("ABCD") == (0, 1, 2, 3)
    assert parse_rank_pattern("B>A>D>C") == (1, 0, 3, 2)
    assert parse_rank_pattern("b, a, d, c") == (1, 0, 3, 2)
    assert format_rank_pattern((1, 0, 3, 2)) == "B>A>D>C"


def test_parse_rank_pattern_rejects_invalid_patterns():
    for pattern in ("", "ABC", "AABC", "ABCE"):
        with pytest.raises(ValueError):
            parse_rank_pattern(pattern)


def test_build_pattern_rows_accepts_exact_pattern_without_ties():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-pattern",
                "completions": [
                    _make_cyclic_completion("A", 4, 0, 0, 3),
                    _make_cyclic_completion("B", 3, 0, 0, 4),
                    _make_cyclic_completion("C", 2, 0, 0, 1),
                    _make_cyclic_completion("D", 1, 0, 0, 2),
                ],
            }
        ]
    )

    rows, stats = build_pattern_rows(
        dataset,
        dimension_pair=("instruction_following", "helpfulness"),
        rank_patterns=("ABCD", "BADC"),
        seed=0,
    )

    assert stats.total_rows == 1
    assert stats.eligible_rows == 1
    assert stats.valid_score_rows == 1
    assert stats.tie_limited_rows == 1
    assert stats.pattern_match_rows == 1
    assert len(rows) == 2
    assert rows[0]["preference_dimension"] == "instruction_following"
    assert rows[0]["responses"] == ["A", "B", "C", "D"]
    assert rows[0]["scores"] == [4.0, 3.0, 2.0, 1.0]
    assert rows[1]["preference_dimension"] == "helpfulness"
    assert rows[1]["responses"] == ["B", "A", "D", "C"]
    assert rows[1]["scores"] == [4.0, 3.0, 2.0, 1.0]


def test_build_pattern_rows_accepts_one_tie_resolved_by_original_placement():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-tie-pattern",
                "completions": [
                    _make_cyclic_completion("A", 4, 0, 0, 3),
                    _make_cyclic_completion("B", 4, 0, 0, 4),
                    _make_cyclic_completion("C", 2, 0, 0, 1),
                    _make_cyclic_completion("D", 1, 0, 0, 3),
                ],
            }
        ]
    )

    rows, stats = build_pattern_rows(
        dataset,
        dimension_pair=("instruction_following", "helpfulness"),
        rank_patterns=("ABCD", "BADC"),
        seed=0,
        max_ties_per_dimension=1,
    )

    assert stats.pattern_match_rows == 1
    assert len(rows) == 2
    assert rows[0]["responses"] == ["A", "B", "C", "D"]
    assert rows[0]["scores"] == [4.0, 4.0, 2.0, 1.0]
    assert rows[1]["responses"] == ["B", "A", "D", "C"]
    assert rows[1]["scores"] == [4.0, 3.0, 3.0, 1.0]


def test_build_pattern_rows_rejects_tie_resolved_to_wrong_pattern():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-wrong-tie",
                "completions": [
                    _make_cyclic_completion("A", 4, 0, 0, 3),
                    _make_cyclic_completion("B", 4, 0, 0, 4),
                    _make_cyclic_completion("C", 2, 0, 0, 1),
                    _make_cyclic_completion("D", 1, 0, 0, 2),
                ],
            }
        ]
    )

    rows, stats = build_pattern_rows(
        dataset,
        dimension_pair=("instruction_following", "helpfulness"),
        rank_patterns=("BACD", "BADC"),
        seed=0,
        max_ties_per_dimension=1,
    )

    assert stats.tie_limited_rows == 1
    assert stats.pattern_match_rows == 0
    assert rows == []


def test_build_pattern_rows_rejects_more_than_one_tie_pair():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-two-ties",
                "completions": [
                    _make_cyclic_completion("A", 4, 0, 0, 3),
                    _make_cyclic_completion("B", 4, 0, 0, 4),
                    _make_cyclic_completion("C", 2, 0, 0, 1),
                    _make_cyclic_completion("D", 2, 0, 0, 2),
                ],
            }
        ]
    )

    rows, stats = build_pattern_rows(
        dataset,
        dimension_pair=("instruction_following", "helpfulness"),
        rank_patterns=("ABCD", "BADC"),
        seed=0,
        max_ties_per_dimension=1,
    )

    assert stats.valid_score_rows == 1
    assert stats.tie_limited_rows == 0
    assert stats.pattern_match_rows == 0
    assert rows == []


def test_build_pattern_rows_rejects_non_matching_dimension():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-non-match",
                "completions": [
                    _make_cyclic_completion("A", 4, 0, 0, 4),
                    _make_cyclic_completion("B", 3, 0, 0, 3),
                    _make_cyclic_completion("C", 2, 0, 0, 2),
                    _make_cyclic_completion("D", 1, 0, 0, 1),
                ],
            }
        ]
    )

    rows, stats = build_pattern_rows(
        dataset,
        dimension_pair=("instruction_following", "helpfulness"),
        rank_patterns=("ABCD", "BADC"),
        seed=0,
    )

    assert stats.pattern_match_rows == 0
    assert rows == []


def test_build_pattern_rows_is_deterministic_for_same_seed():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-many-pattern",
                "completions": [
                    _make_cyclic_completion("X", 0, 0, 0, 0),
                    _make_cyclic_completion("Y", 0, 0, 0, 0),
                    _make_cyclic_completion("A", 4, 0, 0, 3),
                    _make_cyclic_completion("B", 3, 0, 0, 4),
                    _make_cyclic_completion("C", 2, 0, 0, 1),
                    _make_cyclic_completion("D", 1, 0, 0, 2),
                ],
            }
        ]
    )

    rows_1, stats_1 = build_pattern_rows(
        dataset,
        dimension_pair=("instruction_following", "helpfulness"),
        rank_patterns=("ABCD", "BADC"),
        seed=11,
    )
    rows_2, stats_2 = build_pattern_rows(
        dataset,
        dimension_pair=("instruction_following", "helpfulness"),
        rank_patterns=("ABCD", "BADC"),
        seed=11,
    )

    assert rows_1 == rows_2
    assert stats_1 == stats_2


def test_split_rows_is_deterministic_for_same_seed():
    rows = [
        {
            "prompt": f"p{i}",
            "responses": ["A", "B", "C", "D"],
            "scores": [4.0, 3.0, 2.0, 1.0],
            "preference_dimension": "instruction_following",
            "source_index": i,
        }
        for i in range(20)
    ]

    d1 = split_rows_to_dataset_dict(rows, seed=7)
    d2 = split_rows_to_dataset_dict(rows, seed=7)

    assert len(d1["train"]) == 16
    assert len(d1["validation"]) == 2
    assert len(d1["test"]) == 2
    assert d1["train"]["prompt"] == d2["train"]["prompt"]
    assert d1["validation"]["prompt"] == d2["validation"]["prompt"]
    assert d1["test"]["prompt"] == d2["test"]["prompt"]


def test_grouped_split_keeps_source_indices_in_one_split():
    rows = []
    for source_index in range(10):
        for dimension in ("instruction_following", "honesty", "truthfulness", "helpfulness"):
            rows.append(
                {
                    "prompt": f"p{source_index}",
                    "responses": ["A", "B", "C", "D"],
                    "scores": [4.0, 3.0, 2.0, 1.0],
                    "preference_dimension": dimension,
                    "source_index": source_index,
                }
            )

    dataset_dict = split_rows_to_grouped_dataset_dict(rows, seed=7)

    assert len(dataset_dict["train"]) == 32
    assert len(dataset_dict["validation"]) == 4
    assert len(dataset_dict["test"]) == 4

    source_to_split = {}
    for split_name, split_data in dataset_dict.items():
        for source_index in split_data["source_index"]:
            previous = source_to_split.setdefault(source_index, split_name)
            assert previous == split_name


def test_grouped_split_keeps_pattern_rows_together():
    rows = []
    for source_index in range(10):
        for dimension in ("instruction_following", "helpfulness"):
            rows.append(
                {
                    "prompt": f"p{source_index}",
                    "responses": ["A", "B", "C", "D"],
                    "scores": [4.0, 3.0, 2.0, 1.0],
                    "preference_dimension": dimension,
                    "source_index": source_index,
                }
            )

    dataset_dict = split_rows_to_grouped_dataset_dict(rows, seed=7)

    assert len(dataset_dict["train"]) == 16
    assert len(dataset_dict["validation"]) == 2
    assert len(dataset_dict["test"]) == 2

    source_to_split = {}
    for split_name, split_data in dataset_dict.items():
        for source_index in split_data["source_index"]:
            previous = source_to_split.setdefault(source_index, split_name)
            assert previous == split_name
