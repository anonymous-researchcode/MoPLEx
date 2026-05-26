#!/usr/bin/env bash
set -euo pipefail

NUM_PERSONAS="${NUM_PERSONAS:-12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/persona_common.sh"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/persona/config_dpo_qlora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/persona_datasets/persona_${NUM_PERSONAS}_${NEGATIVE_RESPONSE_SOURCE}_pairwise_${PROMPT_MODE}}"
SEEDS="${SEEDS:-42 43}"
MAX_STEPS="${MAX_STEPS:-2000}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-${EVAL_STEPS}}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-1.0}"
DATASET_DOWNSAMPLE_GROUP_KEY="${DATASET_DOWNSAMPLE_GROUP_KEY:-source_index}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing PERSONA pairwise dataset directory: ${DATASET_DIR}" >&2
  echo "Generate it first with:" >&2
  echo "  NUM_PERSONAS=${NUM_PERSONAS} PROMPT_MODE=${PROMPT_MODE} NEGATIVE_RESPONSE_SOURCE=${NEGATIVE_RESPONSE_SOURCE} bash scripts/bash_scripts/evaluate_persona/generate_persona_subset.sh" >&2
  exit 1
fi

if [[ -n "${DIMENSIONS:-}" ]]; then
  read -r -a dimension_args <<< "${DIMENSIONS}"
else
  dimension_args=()
  for ((persona_idx = 0; persona_idx < NUM_PERSONAS; persona_idx++)); do
    dimension_args+=("persona_$(printf "%04d" "${persona_idx}")")
  done
fi

ratio_tag="${DOWNSAMPLE_RATIO//./p}"

for dimension in "${dimension_args[@]}"; do
  dim_tag="${dimension//[^a-zA-Z0-9]/_}"

  for seed in ${SEEDS}; do
    SEED="${seed}"
    run_name="${WANDB_NAME:-qwen3-0.6b-persona${NUM_PERSONAS}-dpo-by-dimension}-${dim_tag}-ds${ratio_tag}-s${seed}"
    output_dir="${OUTPUT_ROOT:-outputs/persona/dpo-by-dimension}/persona${NUM_PERSONAS}-ds${ratio_tag}-${dim_tag}-s${seed}"

    echo "Launching PERSONA DPO for preference_dimension=${dimension}, seed=${seed}, downsample=${DOWNSAMPLE_RATIO}"

    run_persona_dpo_command \
      "${CONFIG_PATH}" \
      "${DATASET_DIR}" \
      "${run_name}" \
      "${output_dir}" \
      --preference_dimensions "${dimension}" \
      --dataset_downsample_ratio "${DOWNSAMPLE_RATIO}" \
      --dataset_downsample_seed "${seed}" \
      --dataset_downsample_group_key "${DATASET_DOWNSAMPLE_GROUP_KEY}" \
      --eval_steps "${EVAL_STEPS}" \
      --save_steps "${SAVE_STEPS}" \
      "$@"
  done
done
