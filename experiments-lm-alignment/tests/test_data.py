# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest
from tempfile import TemporaryDirectory

import pytest
from datasets import Dataset, DatasetDict

from alignment import ScriptArguments, get_dataset
from alignment.data import (
    _maybe_convert_to_listwise,
    _maybe_convert_to_pairwise,
    _maybe_downsample_dataset_dict,
    _to_listwise_dataset,
)


class GetDatasetTest(unittest.TestCase):
    """Test the new get_dataset() method with dataset_mixture API"""

    def test_loading_dataset_mixture(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 0.5},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 0.3,
                },
                {"id": "HuggingFaceH4/testing_codealpaca_small", "columns": ["prompt", "completion"], "weight": 0.2},
            ],
            "seed": 42,
            "test_split_size": 0.1,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # With weights 0.5, 0.3, 0.2 on 100-sample datasets and test_split_size=0.1
        # Total samples = 50 + 30 + 20 = 100
        # Train: 90, Test: 10
        self.assertEqual(len(datasets["train"]), 90)
        self.assertEqual(len(datasets["test"]), 10)

    def test_loading_dataset_mixture_no_test_split(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 0.5},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 0.3,
                },
                {"id": "HuggingFaceH4/testing_codealpaca_small", "columns": ["prompt", "completion"], "weight": 0.2},
            ],
            "seed": 42,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # Total samples = 50 + 30 + 20 = 100 (all in train split)
        self.assertEqual(len(datasets["train"]), 100)
        self.assertNotIn("test", datasets)

    def test_loading_with_unit_weights(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 1.0},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 1.0,
                },
                {"id": "HuggingFaceH4/testing_codealpaca_small", "columns": ["prompt", "completion"], "weight": 1.0},
            ],
            "seed": 42,
            "test_split_size": 0.1,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # Total samples = 100 + 100 + 100 = 300
        # Train: 270, Test: 30
        self.assertEqual(len(datasets["train"]), 270)
        self.assertEqual(len(datasets["test"]), 30)

    def test_loading_with_fractional_weights(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 0.7},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 0.4,
                },
            ],
            "seed": 42,
            "test_split_size": 0.1,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # Total samples = 70 + 40 = 110
        # Train: 99, Test: 11
        self.assertEqual(len(datasets["train"]), 99)
        self.assertEqual(len(datasets["test"]), 11)

    def test_loading_fails_with_invalid_dataset_mixture(self):
        # Test that invalid dataset_mixture configuration raises error
        with pytest.raises(ValueError, match=r"'datasets' must be a list"):
            _ = ScriptArguments(dataset_mixture={"datasets": "invalid"})

        with pytest.raises(ValueError, match=r"dataset_mixture must be a dictionary"):
            _ = ScriptArguments(dataset_mixture="invalid")

    def test_loading_single_dataset(self):
        # Test loading a single dataset using dataset_name instead of dataset_mixture
        args = ScriptArguments(dataset_name="HuggingFaceH4/testing_alpaca_small")
        datasets = get_dataset(args)
        # Single dataset should have both train and test splits
        self.assertIn("train", datasets)
        self.assertEqual(len(datasets["train"]), 100)
        self.assertIn("test", datasets)
        self.assertEqual(len(datasets["test"]), 100)

    def test_loading_local_dataset_mixture_entry(self):
        with TemporaryDirectory() as tmpdir:
            local_ds = DatasetDict(
                {
                    "train": Dataset.from_list([{"prompt": "p1", "completion": "c1"}, {"prompt": "p2", "completion": "c2"}]),
                    "test": Dataset.from_list([{"prompt": "p3", "completion": "c3"}]),
                }
            )
            local_ds.save_to_disk(tmpdir)

            dataset_mixture = {
                "datasets": [
                    {"id": tmpdir, "split": "train", "columns": ["prompt", "completion"], "weight": 1.0},
                ],
                "seed": 42,
            }

            args = ScriptArguments(dataset_mixture=dataset_mixture)
            datasets = get_dataset(args)
            self.assertEqual(len(datasets["train"]), 2)
            self.assertNotIn("test", datasets)

    def test_listwise_conversion_sorts_by_dimension(self):
        data = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "completions": [
                        {"response": "A", "scores": {"helpfulness": 1.0}},
                        {"response": "B", "scores": {"helpfulness": 3.0}},
                        {"response": "C", "scores": {"helpfulness": 2.0}},
                        {"response": "D", "scores": {"helpfulness": 0.5}},
                    ],
                }
            ]
        )
        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["helpfulness"],
            listwise_num_responses=4,
        )
        converted = _to_listwise_dataset(data, args)
        self.assertEqual(len(converted), 1)
        first = converted[0]
        self.assertEqual(first["responses"], ["B", "C", "A", "D"])

    def test_listwise_requires_dimension(self):
        with pytest.raises(ValueError, match=r"preference_dimensions"):
            _ = ScriptArguments(dataset_name="dummy", dataset_format="listwise")

    def test_listwise_expands_all_subrankings(self):
        data = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "completions": [
                        {"response": "A", "scores": {"helpfulness": 4.0}},
                        {"response": "B", "scores": {"helpfulness": 3.0}},
                        {"response": "C", "scores": {"helpfulness": 2.0}},
                        {"response": "D", "scores": {"helpfulness": 1.0}},
                    ],
                }
            ]
        )

        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["helpfulness"],
            listwise_num_responses=2,
        )

        converted = _to_listwise_dataset(data, args)
        self.assertEqual(len(converted), 6)

        response_pairs = {tuple(item) for item in converted["responses"]}
        expected_pairs = {
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
        }
        self.assertEqual(response_pairs, expected_pairs)

    def test_listwise_cyclic_filter_integration(self):
        data = Dataset.from_list(
            [
                {
                    "instruction": "p1",
                    "completions": [
                        {
                            "response": "A",
                            "annotations": {
                                "instruction_following": {"Rating": 4.0},
                                "helpfulness": {"Rating": 1.0},
                            },
                        },
                        {
                            "response": "B",
                            "annotations": {
                                "instruction_following": {"Rating": 3.0},
                                "helpfulness": {"Rating": 4.0},
                            },
                        },
                        {
                            "response": "C",
                            "annotations": {
                                "instruction_following": {"Rating": 2.0},
                                "helpfulness": {"Rating": 3.0},
                            },
                        },
                        {
                            "response": "D",
                            "annotations": {
                                "instruction_following": {"Rating": 1.0},
                                "helpfulness": {"Rating": 2.0},
                            },
                        },
                    ],
                }
            ]
        )

        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["instruction_following", "helpfulness"],
            listwise_use_cyclic_filter=True,
            listwise_prompt_column="instruction",
            listwise_responses_column="completions",
            listwise_response_text_key="response",
            listwise_annotations_key="annotations",
        )

        converted = _to_listwise_dataset(data, args)
        self.assertEqual(len(converted), 2)
        pref_dims = set(converted["preference_dimension"])
        self.assertEqual(pref_dims, {"instruction_following", "helpfulness"})

    def test_preformatted_listwise_split_skips_reconversion(self):
        preformatted = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "responses": ["A", "B", "C", "D"],
                    "scores": [4.0, 3.0, 2.0, 1.0],
                    "preference_dimension": "instruction_following",
                    "source_index": 0,
                },
                {
                    "prompt": "p1",
                    "responses": ["B", "C", "D", "A"],
                    "scores": [4.0, 3.0, 2.0, 1.0],
                    "preference_dimension": "helpfulness",
                    "source_index": 0,
                },
            ]
        )
        ds_dict = DatasetDict({"train": preformatted})

        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["helpfulness"],
        )

        converted = _maybe_convert_to_listwise(ds_dict, args)
        self.assertEqual(len(converted["train"]), 1)
        self.assertEqual(set(converted["train"]["preference_dimension"]), {"helpfulness"})

    def test_preformatted_pairwise_split_filters_by_preference_dimension(self):
        preformatted = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "chosen": "A",
                    "rejected": "B",
                    "preference_dimension": "instruction_following",
                },
                {
                    "prompt": "p2",
                    "chosen": "C",
                    "rejected": "D",
                    "preference_dimension": "helpfulness",
                },
            ]
        )
        ds_dict = DatasetDict({"train": preformatted})

        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="pairwise",
            preference_dimensions=["helpfulness"],
        )

        converted = _maybe_convert_to_pairwise(ds_dict, args)
        self.assertEqual(len(converted["train"]), 1)
        self.assertEqual(converted["train"][0]["preference_dimension"], "helpfulness")

    def test_preformatted_listwise_split_reduces_with_all_subrankings(self):
        preformatted = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "responses": ["A", "B", "C", "D"],
                    "scores": [4.0, 3.0, 2.0, 1.0],
                    "preference_dimension": "instruction_following",
                }
            ]
        )
        ds_dict = DatasetDict({"train": preformatted})

        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["instruction_following", "helpfulness"],
            listwise_num_responses=2,
            listwise_min_responses=2,
        )

        converted = _maybe_convert_to_listwise(ds_dict, args)
        train = converted["train"]
        self.assertEqual(len(train), 6)

        response_pairs = {tuple(item) for item in train["responses"]}
        expected_pairs = {
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
        }
        self.assertEqual(response_pairs, expected_pairs)

    def test_preformatted_listwise_split_reduces_to_triplets_and_full_rankings(self):
        preformatted = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "responses": ["A", "B", "C", "D"],
                    "scores": [4.0, 3.0, 2.0, 1.0],
                    "preference_dimension": "instruction_following",
                }
            ]
        )

        triplet_args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["instruction_following"],
            listwise_num_responses=3,
            listwise_min_responses=2,
        )
        triplets = _maybe_convert_to_listwise(DatasetDict({"train": preformatted}), triplet_args)["train"]
        self.assertEqual(len(triplets), 4)
        self.assertEqual({len(responses) for responses in triplets["responses"]}, {3})

        full_args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["instruction_following"],
            listwise_num_responses=4,
            listwise_min_responses=2,
        )
        full = _maybe_convert_to_listwise(DatasetDict({"train": preformatted}), full_args)["train"]
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0]["responses"], ["A", "B", "C", "D"])

    def test_grouped_downsampling_keeps_source_index_rows_together(self):
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

        args = ScriptArguments(
            dataset_name="dummy",
            dataset_downsample_ratio=0.5,
            dataset_downsample_seed=13,
        )
        downsampled = _maybe_downsample_dataset_dict(DatasetDict({"train": Dataset.from_list(rows)}), args)["train"]

        self.assertEqual(len(downsampled), 10)
        counts = {}
        for source_index in downsampled["source_index"]:
            counts[source_index] = counts.get(source_index, 0) + 1
        self.assertEqual(len(counts), 5)
        self.assertEqual(set(counts.values()), {2})

    def test_downsampling_applies_to_train_validation_and_test_deterministically(self):
        dataset_dict = DatasetDict(
            {
                split: Dataset.from_list([{"prompt": f"{split}-{idx}", "completion": f"c{idx}"} for idx in range(10)])
                for split in ("train", "validation", "test")
            }
        )
        args = ScriptArguments(
            dataset_name="dummy",
            dataset_downsample_ratio=0.5,
            dataset_downsample_seed=7,
        )

        first = _maybe_downsample_dataset_dict(dataset_dict, args)
        second = _maybe_downsample_dataset_dict(dataset_dict, args)

        self.assertEqual({split: len(first[split]) for split in first}, {"train": 5, "validation": 5, "test": 5})
        self.assertEqual(first["train"]["prompt"], second["train"]["prompt"])
        self.assertEqual(first["validation"]["prompt"], second["validation"]["prompt"])
        self.assertEqual(first["test"]["prompt"], second["test"]["prompt"])

    def test_downsampling_uses_row_fallback_without_group_key(self):
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(
                    [{"prompt": f"p{idx}", "completion": f"c{idx}"} for idx in range(10)]
                )
            }
        )
        args = ScriptArguments(
            dataset_name="dummy",
            dataset_downsample_ratio=0.3,
            dataset_downsample_seed=5,
        )

        downsampled = _maybe_downsample_dataset_dict(dataset_dict, args)
        self.assertEqual(len(downsampled["train"]), 3)

    def test_listwise_multiple_dimensions_non_cyclic(self):
        data = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "completions": [
                        {
                            "response": "A",
                            "scores": {"helpfulness": 1.0, "honesty": 4.0},
                        },
                        {
                            "response": "B",
                            "scores": {"helpfulness": 3.0, "honesty": 3.0},
                        },
                        {
                            "response": "C",
                            "scores": {"helpfulness": 2.0, "honesty": 2.0},
                        },
                        {
                            "response": "D",
                            "scores": {"helpfulness": 0.5, "honesty": 1.0},
                        },
                    ],
                }
            ]
        )
        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["helpfulness", "honesty"],
            listwise_num_responses=4,
        )
        converted = _to_listwise_dataset(data, args)
        self.assertEqual(len(converted), 2)
        self.assertEqual(set(converted["preference_dimension"]), {"helpfulness", "honesty"})

    def test_augmented_listwise_reduction_selects_original_prefix_and_generated_tail(self):
        preformatted = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "responses": ["A", "B", "C", "D", "G1", "G2"],
                    "scores": [4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
                    "preference_dimension": "helpfulness",
                    "source_index": 0,
                    "ranked_prefix_length": 4,
                }
            ]
        )
        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["helpfulness"],
            listwise_num_responses=2,
            listwise_num_generated_responses=2,
            listwise_min_responses=2,
        )

        converted = _maybe_convert_to_listwise(DatasetDict({"train": preformatted}), args)["train"]

        self.assertEqual(len(converted), 6)
        self.assertEqual({len(responses) for responses in converted["responses"]}, {4})
        self.assertEqual(set(converted["ranked_prefix_length"]), {2})
        self.assertTrue(all(responses[2:] == ["G1", "G2"] for responses in converted["responses"]))
        expected_original_pairs = {
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
        }
        observed_original_pairs = {tuple(responses[:2]) for responses in converted["responses"]}
        self.assertEqual(observed_original_pairs, expected_original_pairs)

    def test_augmented_listwise_reduction_falls_back_for_unaugmented_rows(self):
        preformatted = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "responses": ["A", "B", "C", "D"],
                    "scores": [4.0, 3.0, 2.0, 1.0],
                    "preference_dimension": "helpfulness",
                    "source_index": 0,
                }
            ]
        )
        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimensions=["helpfulness"],
            listwise_num_responses=2,
            listwise_num_generated_responses=2,
            listwise_min_responses=2,
        )

        converted = _maybe_convert_to_listwise(DatasetDict({"validation": preformatted}), args)["validation"]

        self.assertEqual(len(converted), 6)
        self.assertEqual({len(responses) for responses in converted["responses"]}, {2})
        self.assertEqual(set(converted["ranked_prefix_length"]), {2})
        response_pairs = {tuple(item) for item in converted["responses"]}
        expected_pairs = {
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
        }
        self.assertEqual(response_pairs, expected_pairs)
