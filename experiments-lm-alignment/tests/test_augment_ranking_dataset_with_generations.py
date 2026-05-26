import importlib.util
from pathlib import Path
import sys

from datasets import Dataset, DatasetDict, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from alignment import ScriptArguments
from alignment.data import _maybe_convert_to_listwise, _maybe_downsample_dataset_dict


SCRIPT_PATH = REPO_ROOT / "scripts" / "augment_ranking_dataset_with_generations.py"
spec = importlib.util.spec_from_file_location("augment_ranking_dataset_with_generations", SCRIPT_PATH)
augment_script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = augment_script
spec.loader.exec_module(augment_script)


def make_dataset_dict():
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "prompt": "prompt one",
                        "responses": ["A", "B"],
                        "scores": [2.0, 1.0],
                        "preference_dimension": "helpfulness",
                        "source_index": 7,
                        "kept_metadata": "first",
                    },
                    {
                        "prompt": "prompt one",
                        "responses": ["B", "A"],
                        "scores": [3.0, 0.0],
                        "preference_dimension": "honesty",
                        "source_index": 7,
                        "kept_metadata": "second",
                    },
                    {
                        "prompt": "prompt two",
                        "responses": ["C", "D"],
                        "scores": [5.0, 4.0],
                        "preference_dimension": "helpfulness",
                        "source_index": 8,
                        "kept_metadata": "third",
                    },
                ]
            ),
            "validation": Dataset.from_list(
                [
                    {
                        "prompt": "validation prompt",
                        "responses": ["V1", "V2"],
                        "scores": [1.0, 0.0],
                        "preference_dimension": "helpfulness",
                        "source_index": 0,
                        "kept_metadata": "validation",
                    }
                ]
            ),
        }
    )


def test_augment_appends_generated_tail_and_reuses_source_index_group():
    prompts_seen = []

    def fake_generate(prompts):
        prompts_seen.extend(prompts)
        return [[f"{prompt} generated 1", f"{prompt} generated 2"] for prompt in prompts]

    augmented, stats = augment_script.augment_dataset_dict(
        make_dataset_dict(),
        fake_generate,
        splits=["train"],
        max_rows=None,
        batch_size=2,
        num_new_responses=2,
        deduplicate=True,
        allow_short_generation=False,
        augmentation_model="fake/model",
        generation_seed=123,
    )

    assert prompts_seen == ["prompt one", "prompt two"]
    assert stats["prompt_groups"] == 2
    assert stats["generated_responses"] == 4

    first = augmented["train"][0]
    second = augmented["train"][1]
    third = augmented["train"][2]

    assert first["responses"][:2] == ["A", "B"]
    assert first["scores"][:2] == [2.0, 1.0]
    assert first["responses"][2:] == ["prompt one generated 1", "prompt one generated 2"]
    assert first["scores"][2:] == [0.0, 0.0]
    assert first["ranked_prefix_length"] == 2
    assert first["augmentation_model"] == "fake/model"
    assert first["num_generated_responses"] == 2
    assert first["generation_seed"] == 123
    assert first["generation_prompt_key"] == "train:source_index:7"
    assert first["kept_metadata"] == "first"

    assert second["responses"][:2] == ["B", "A"]
    assert second["responses"][2:] == first["responses"][2:]
    assert second["generation_prompt_key"] == first["generation_prompt_key"]
    assert second["kept_metadata"] == "second"

    assert third["responses"][2:] == ["prompt two generated 1", "prompt two generated 2"]
    assert third["generation_prompt_key"] == "train:source_index:8"


def test_augment_defaults_to_train_only_and_leaves_validation_ungenerated():
    prompts_seen = []

    def fake_generate(prompts):
        prompts_seen.extend(prompts)
        return [[f"{prompt} generated"] for prompt in prompts]

    augmented, stats = augment_script.augment_dataset_dict(
        make_dataset_dict(),
        fake_generate,
        splits=["train", "validation"],
        max_rows=None,
        batch_size=2,
        num_new_responses=1,
        deduplicate=True,
        allow_short_generation=False,
        augmentation_model="fake/model",
        generation_seed=123,
    )

    assert prompts_seen == ["prompt one", "prompt two"]
    assert stats["augment_splits"] == ["train"]
    assert augmented["train"][0]["responses"] == ["A", "B", "prompt one generated"]
    assert augmented["validation"][0]["responses"] == ["V1", "V2"]
    assert augmented["validation"][0]["scores"] == [1.0, 0.0]
    assert augmented["validation"][0]["ranked_prefix_length"] == 2
    assert augmented["validation"][0]["num_generated_responses"] == 0


def test_augment_deduplicates_generated_responses_against_group_originals():
    def fake_generate(prompts):
        assert prompts == ["prompt one"]
        return [["A", "B", "fresh one", "fresh one", "fresh two"]]

    augmented, stats = augment_script.augment_dataset_dict(
        make_dataset_dict(),
        fake_generate,
        splits=["train"],
        max_rows=2,
        batch_size=1,
        num_new_responses=2,
        deduplicate=True,
        allow_short_generation=False,
        augmentation_model="fake/model",
        generation_seed=1,
    )

    assert augmented["train"][0]["responses"][2:] == ["fresh one", "fresh two"]
    assert stats["dropped_generated_responses"] == 3


def test_save_augmented_dataset_reloads_from_disk(tmp_path):
    def fake_generate(prompts):
        return [[f"{prompt} generated"] for prompt in prompts]

    augmented, stats = augment_script.augment_dataset_dict(
        make_dataset_dict(),
        fake_generate,
        splits=["train"],
        max_rows=1,
        batch_size=1,
        num_new_responses=1,
        deduplicate=True,
        allow_short_generation=False,
        augmentation_model="fake/model",
        generation_seed=42,
    )

    output_dir = tmp_path / "augmented"
    augment_script.save_augmented_dataset(augmented, str(output_dir), stats, overwrite=False)

    reloaded = load_from_disk(str(output_dir))
    assert len(reloaded["train"]) == 1
    assert reloaded["train"][0]["responses"] == ["A", "B", "prompt one generated"]
    assert (output_dir / "augmentation_stats.json").exists()


def test_group_downsampling_applies_before_train_generation():
    rows = []
    for source_index in range(10):
        for dimension in ("helpfulness", "honesty"):
            rows.append(
                {
                    "prompt": f"prompt {source_index}",
                    "responses": ["A", "B"],
                    "scores": [1.0, 0.0],
                    "preference_dimension": dimension,
                    "source_index": source_index,
                }
            )
    dataset_dict = DatasetDict({"train": Dataset.from_list(rows)})
    prompts_seen = []

    def fake_generate(prompts):
        prompts_seen.extend(prompts)
        return [[f"{prompt} generated"] for prompt in prompts]

    augmented, stats = augment_script.augment_dataset_dict(
        dataset_dict,
        fake_generate,
        splits=["train"],
        augment_splits=["train"],
        max_rows=None,
        dataset_downsample_ratio=0.3,
        dataset_downsample_seed=42,
        dataset_downsample_group_key="source_index",
        dataset_downsample_splits=["train"],
        batch_size=4,
        num_new_responses=1,
        deduplicate=True,
        allow_short_generation=False,
        augmentation_model="fake/model",
        generation_seed=42,
    )

    assert len(prompts_seen) == 3
    assert len(augmented["train"]) == 6
    assert len(set(augmented["train"]["source_index"])) == 3
    expected = _maybe_downsample_dataset_dict(
        dataset_dict,
        ScriptArguments(
            dataset_name="dummy",
            dataset_downsample_ratio=0.3,
            dataset_downsample_seed=42,
            dataset_downsample_group_key="source_index",
            dataset_downsample_splits=["train"],
        ),
    )["train"]
    assert augmented["train"]["source_index"] == expected["source_index"]
    assert stats["dataset_downsample_ratio"] == 0.3
    assert stats["dataset_downsample_group_key"] == "source_index"


def test_augmented_partial_tail_flows_through_listwise_reduction():
    augmented = Dataset.from_list(
        [
            {
                "prompt": "p",
                "responses": ["A", "B", "G1", "G2"],
                "scores": [2.0, 1.0, 0.0, 0.0],
                "preference_dimension": "helpfulness",
                "source_index": 0,
                "ranked_prefix_length": 2,
            }
        ]
    )
    args = ScriptArguments(
        dataset_name="dummy",
        dataset_format="listwise",
        preference_dimensions=["helpfulness"],
        listwise_num_responses=3,
        listwise_min_responses=2,
    )

    converted = _maybe_convert_to_listwise(DatasetDict({"train": augmented}), args)["train"]

    assert len(converted) == 2
    assert {tuple(row) for row in converted["responses"]} == {("A", "B", "G1"), ("A", "B", "G2")}
    assert set(converted["ranked_prefix_length"]) == {2}
