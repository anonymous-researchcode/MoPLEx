import argparse
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_persona_baselines_dataset.py"
spec = importlib.util.spec_from_file_location("prepare_persona_baselines_dataset", SCRIPT_PATH)
persona_prepare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = persona_prepare
spec.loader.exec_module(persona_prepare)


def make_args(**overrides):
    defaults = dict(
        train_ratio=0.0,
        validation_ratio=0.0,
        test_ratio=1.0,
        split_seed=42,
        split_mode="row_by_persona",
        hard_negative_seed=123,
        num_responses_per_prompt=4,
        eval_num_responses_per_prompt=None,
        negative_pool="same_split",
        negative_response_source="original",
        hard_negative_persona_pool="all",
        allow_short_listwise=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_raw_rows():
    rows = []
    for idx, persona_id in enumerate(["p0", "p1", "p2", "p3"]):
        rows.append(
            {
                "source_split": "train",
                "source_index": idx,
                "global_index": idx,
                "row_id": f"train:{idx}",
                "persona_id": persona_id,
                "persona_text": f"persona text {idx}",
                "instruction": "What is the meaning of life?",
                "chosen": f"personalized answer {idx}",
                "rejected": "generic answer",
            }
        )
    return rows


def make_prompt_overlap_rows(num_prompts=12, num_personas=3):
    rows = []
    idx = 0
    for prompt_idx in range(num_prompts):
        for persona_idx in range(num_personas):
            rows.append(
                {
                    "source_split": "train",
                    "source_index": idx,
                    "global_index": idx,
                    "row_id": f"train:{idx}",
                    "persona_id": f"p{persona_idx}",
                    "persona_text": f"persona text {persona_idx}",
                    "instruction": f"shared prompt {prompt_idx}",
                    "chosen": f"personalized answer {prompt_idx}-{persona_idx}",
                    "rejected": f"generic answer {prompt_idx}",
                }
            )
            idx += 1
    return rows


def make_split_prompt_rows():
    rows = []
    idx = 0
    for split in ["train", "validation", "test"]:
        for persona_id in ["p0", "p1", "p2", "p3"]:
            rows.append(
                {
                    "source_split": "train",
                    "source_index": idx,
                    "global_index": idx,
                    "row_id": f"train:{idx}",
                    "persona_id": persona_id,
                    "persona_text": f"persona text {persona_id}",
                    "instruction": f"shared prompt {split}",
                    "chosen": f"personalized {split} {persona_id}",
                    "rejected": f"generic answer {split}",
                }
            )
            idx += 1
    return rows


def split_map_for_split_prompt_rows(raw_rows):
    row_to_split = {}
    for row in raw_rows:
        row_to_split[row["row_id"]] = row["instruction"].replace("shared prompt ", "")
    return row_to_split


def test_prompt_split_keeps_each_instruction_in_one_split():
    args = make_args(train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25, split_mode="prompt")
    raw_rows = make_prompt_overlap_rows()

    row_to_split = persona_prepare.split_rows(raw_rows, args)

    prompt_to_splits = {}
    for row in raw_rows:
        prompt_to_splits.setdefault(row["instruction"], set()).add(row_to_split[row["row_id"]])

    assert all(len(splits) == 1 for splits in prompt_to_splits.values())
    assert {next(iter(splits)) for splits in prompt_to_splits.values()} == {
        "train",
        "validation",
        "test",
    }


def test_pairwise_mapping_preserves_persona_metadata():
    args = make_args()
    rows = persona_prepare.build_persona_rows(make_raw_rows(), ["p0", "p1", "p2"], args)

    split_rows, stats = persona_prepare.build_pairwise_rows(rows, "instruction_only", args)
    first = split_rows["test"][0]

    assert stats["kept"] == 3
    assert first["prompt"] == "What is the meaning of life?"
    assert first["chosen"] == "personalized answer 0"
    assert first["rejected"] == "generic answer"
    assert first["persona_id"] == "p0"
    assert first["preference_dimension"] == "persona_0000"
    assert first["negative_response_source"] == "original"
    assert first["rejected_persona_id"] == "generic_original"


def test_pairwise_can_use_other_persona_negative():
    args = make_args(negative_response_source="other_persona")
    rows = persona_prepare.build_persona_rows(make_raw_rows(), ["p0", "p1", "p2"], args)

    split_rows, stats = persona_prepare.build_pairwise_rows(rows, "instruction_only", args)
    first = split_rows["test"][0]

    assert stats["kept"] == 3
    assert first["chosen"] == "personalized answer 0"
    assert first["rejected"] in {"personalized answer 1", "personalized answer 2"}
    assert first["rejected"] != "generic answer"
    assert first["negative_response_source"] == "other_persona"
    assert first["rejected_persona_id"] in {"p1", "p2"}


def test_listwise_rows_put_target_first_and_sample_same_prompt_negatives():
    args = make_args(num_responses_per_prompt=4)
    rows = persona_prepare.build_persona_rows(make_raw_rows(), ["p0", "p1", "p2"], args)

    split_rows, stats = persona_prepare.build_listwise_rows(rows, "instruction_only", args)
    first = split_rows["test"][0]

    assert stats["kept"] == 3
    assert first["responses"][0] == "personalized answer 0"
    assert first["responses"][1] == "generic answer"
    assert set(first["responses"][2:]) == {"personalized answer 1", "personalized answer 2"}
    assert first["ranked_prefix_length"] == 1
    assert first["scores"] == [1.0, 0.0, 0.0, 0.0]
    assert "p0" not in first["negative_persona_ids"]


def test_listwise_num_responses_controls_k():
    args = make_args(num_responses_per_prompt=3)
    rows = persona_prepare.build_persona_rows(make_raw_rows(), ["p0", "p1", "p2"], args)

    split_rows, _ = persona_prepare.build_listwise_rows(rows, "instruction_only", args)

    assert all(len(row["responses"]) == 3 for row in split_rows["test"])


def test_listwise_can_skip_original_and_use_only_other_persona_negatives():
    args = make_args(num_responses_per_prompt=3, negative_response_source="other_persona")
    rows = persona_prepare.build_persona_rows(make_raw_rows(), ["p0", "p1", "p2"], args)

    split_rows, stats = persona_prepare.build_listwise_rows(rows, "instruction_only", args)
    first = split_rows["test"][0]

    assert stats["kept"] == 3
    assert len(first["responses"]) == 3
    assert first["responses"][0] == "personalized answer 0"
    assert "generic answer" not in first["responses"]
    assert first["negative_response_source"] == "other_persona"
    assert set(first["negative_persona_ids"]).issubset({"p1", "p2"})


def test_listwise_eval_k_applies_only_to_validation_and_test():
    args = make_args(num_responses_per_prompt=4, eval_num_responses_per_prompt=2, negative_response_source="other_persona")
    raw_rows = make_split_prompt_rows()
    rows = persona_prepare.build_persona_rows(
        raw_rows,
        ["p0", "p1", "p2", "p3"],
        args,
        row_to_split=split_map_for_split_prompt_rows(raw_rows),
    )

    split_rows, stats = persona_prepare.build_listwise_rows(rows, "instruction_only", args)

    assert stats["kept"] == 12
    assert all(len(row["responses"]) == 4 for row in split_rows["train"])
    assert all(len(row["responses"]) == 2 for row in split_rows["validation"])
    assert all(len(row["responses"]) == 2 for row in split_rows["test"])
    assert all("generic_original" not in row["negative_persona_ids"] for row in split_rows["validation"])
    assert all("generic_original" not in row["negative_persona_ids"] for row in split_rows["test"])
    for split in ["validation", "test"]:
        for row in split_rows[split]:
            assert row["negative_persona_ids"][0] != row["persona_id"]
            assert row["responses"][1].startswith(f"personalized {split} ")


def test_listwise_dir_name_includes_eval_k_only_when_different():
    base_args = make_args(num_responses_per_prompt=4)
    same_args = make_args(num_responses_per_prompt=4, eval_num_responses_per_prompt=4)
    eval_args = make_args(num_responses_per_prompt=4, eval_num_responses_per_prompt=2)

    assert persona_prepare.listwise_dir_name("persona_10", "persona_conditioned", base_args).endswith(
        "_top1_listwise_persona_conditioned_k4"
    )
    assert persona_prepare.listwise_dir_name("persona_10", "persona_conditioned", same_args).endswith(
        "_top1_listwise_persona_conditioned_k4"
    )
    assert persona_prepare.listwise_dir_name("persona_10", "persona_conditioned", eval_args).endswith(
        "_top1_listwise_persona_conditioned_k4_evalk2"
    )


def test_listwise_can_sample_negatives_from_unselected_personas():
    args = make_args(num_responses_per_prompt=4)
    raw_rows = make_raw_rows()
    row_to_split = persona_prepare.split_rows_by_persona(raw_rows, args)
    dimension_by_persona = persona_prepare.persona_dimension_map(["p0"])
    target_rows = persona_prepare.build_persona_rows(
        raw_rows,
        ["p0"],
        args,
        row_to_split=row_to_split,
        dimension_by_persona=dimension_by_persona,
    )
    hard_negative_rows = persona_prepare.build_persona_rows(
        raw_rows,
        ["p0", "p1", "p2", "p3"],
        args,
        row_to_split=row_to_split,
        dimension_by_persona=dimension_by_persona,
    )

    split_rows, stats = persona_prepare.build_listwise_rows(
        target_rows,
        "instruction_only",
        args,
        hard_negative_rows=hard_negative_rows,
    )
    first = split_rows["test"][0]

    assert stats["kept"] == 1
    assert len(first["responses"]) == 4
    assert set(first["negative_persona_ids"][1:]).issubset({"p1", "p2", "p3"})
