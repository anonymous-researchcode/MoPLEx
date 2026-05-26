#!/usr/bin/env python
"""Inspect raw UltraFeedback and listwise conversions for four dimensions.

Usage:
	PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python tests/inspect_ultrafeedback_listwise.py
"""
# %%
from __future__ import annotations
from dataclasses import replace
import datasets
import sys
from pathlib import Path

try:
	from alignment import ScriptArguments, get_dataset
except ModuleNotFoundError:
	# Support running this file directly from notebooks or arbitrary CWDs.
	project_root = Path(__file__).resolve().parents[1]
	src_dir = project_root / "src"
	if str(src_dir) not in sys.path:
		sys.path.insert(0, str(src_dir))
	from alignment import ScriptArguments, get_dataset


RAW_DATASET_ID ="HuggingFaceH4/ultrafeedback_binarized" # "openbmb/UltraFeedback"

DIMENSION_CANDIDATES = {
	"helpfulness": ["helpfulness"],
	"honesty": ["honesty"],
	"instruction_following": ["instruction_following", "instruction-following"],
	"truthfulness": ["truthfulness"],
}


def _choose_existing(columns: list[str], candidates: list[str]) -> str | None:
	for name in candidates:
		if name in columns:
			return name
	return None


def _pick_prompt_column(columns: list[str]) -> str:
	prompt_col = _choose_existing(columns, ["prompt", "instruction", "question", "query"])
	if prompt_col is None:
		raise ValueError(
			f"Could not detect prompt column. Available columns: {columns}. "
			"Try extending candidate names in this script."
		)
	return prompt_col


def _pick_responses_column(columns: list[str]) -> str:
	responses_col = _choose_existing(columns, ["completions", "responses", "candidates"])
	if responses_col is None:
		raise ValueError(
			f"Could not detect responses column. Available columns: {columns}. "
			"Try extending candidate names in this script."
		)
	return responses_col


def _print_raw_summary(raw_ds: datasets.DatasetDict) -> None:
	print("=" * 100)
	print(f"Raw dataset: {RAW_DATASET_ID}")
	print(raw_ds)
	for split_name, split in raw_ds.items():
		print(f"  - split={split_name} num_rows={len(split)}")
		print(f"    columns={split.column_names}")
	print("=" * 100)


def _print_listwise_preview(dimension: str, ds_dict: datasets.DatasetDict) -> None:
	print("-" * 100)
	print(f"Listwise dimension: {dimension}")
	print(ds_dict)
	for split_name, split in ds_dict.items():
		print(f"  - split={split_name} num_rows={len(split)}")
		if len(split) == 0:
			continue
		row = split[0]
		responses = row["responses"]
		scores = row["scores"]
		print(f"    prompt={row['prompt'][:160]!r}")
		print(f"    num_ranked_responses={len(responses)}")
		for idx, (resp, score) in enumerate(zip(responses, scores), start=1):
			print(f"      {idx}. score={score:.4f} text={resp[:120]!r}")


def _build_base_script_args(prompt_col: str, responses_col: str) -> ScriptArguments:
	return ScriptArguments(
		dataset_name=RAW_DATASET_ID,
		dataset_format="pairwise",
		preference_dimension="helpfulness",
		listwise_num_responses=4,
		listwise_min_responses=2,
		listwise_prompt_column=prompt_col,
		listwise_responses_column=responses_col,
		listwise_response_text_key="response",
		listwise_scores_key="scores",
		listwise_annotations_key="annotations",
	)


def _load_dimension_dataset(base_args: ScriptArguments, logical_dimension: str) -> tuple[str, datasets.DatasetDict]:
    candidates = DIMENSION_CANDIDATES[logical_dimension]
    # last_error: Exception | None = None
    # for dim_key in candidates:
    # try:
    dim_key = candidates[0]
    args = replace(base_args, preference_dimension=dim_key)
    ds = get_dataset(args)
    return dim_key, ds
    # except Exception as exc:  # noqa: BLE001 - inspection utility should keep trying fallbacks
    # 	last_error = exc

    # raise RuntimeError(
    # 	f"Failed to build listwise dataset for logical dimension '{logical_dimension}' "
    # 	f"with candidates {candidates}. Last error: {last_error}"
    # )


# %%
# def main() -> None:
raw_ds = datasets.load_dataset(RAW_DATASET_ID)
_print_raw_summary(raw_ds)

# Use the first available split for schema inspection.
first_split_name = list(raw_ds.keys())[0]
first_split = raw_ds[first_split_name]
columns = first_split.column_names

# prompt_col = _pick_prompt_column(columns)
# responses_col = _pick_responses_column(columns)
# print(f"Detected prompt column: {prompt_col}")
# print(f"Detected responses column: {responses_col}")


# %%
base_args = _build_base_script_args(prompt_col=prompt_col, responses_col=responses_col)

# %%
for logical_dimension in ["helpfulness", "honesty", "instruction_following", "truthfulness"]:
    resolved_key, ds = _load_dimension_dataset(base_args=base_args, logical_dimension=logical_dimension)
    print(f"Using score key '{resolved_key}' for logical dimension '{logical_dimension}'")
    _print_listwise_preview(logical_dimension, ds)

print("=" * 100)
print("Inspection complete.")


# if __name__ == "__main__":
# 	main()
