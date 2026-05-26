#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement_if_truthfulness_allow_one_tie}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
BASE_NUM_RESPONSES="${BASE_NUM_RESPONSES:-4}"
NUM_NEW_RESPONSES="${NUM_NEW_RESPONSES:-2}"
TOTAL_RESPONSES_TAG="$((BASE_NUM_RESPONSES + NUM_NEW_RESPONSES))"
SEED="${SEED:-42}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-0.2}"
DOWNSAMPLE_GROUP_KEY="${DOWNSAMPLE_GROUP_KEY:-source_index}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-2.0}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

dataset_tag="$(basename "${DATASET_DIR}")"
ratio_tag="${DOWNSAMPLE_RATIO//./p}"
model_tag="$(basename "${MODEL_NAME_OR_PATH}")"
model_tag="${model_tag//./p}"
model_tag="${model_tag//-/_}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/${dataset_tag}_train_augmented_${model_tag}_ds${ratio_tag}_k${TOTAL_RESPONSES_TAG}_s${SEED}}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing ranking dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

overwrite_args=()
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  overwrite_args=(--overwrite)
fi

max_rows_args=()
if [[ -n "${MAX_ROWS:-}" ]]; then
  max_rows_args=(--max_rows "${MAX_ROWS}")
fi

deduplicate_args=()
if [[ "${DEDUPLICATE:-true}" == "false" ]]; then
  deduplicate_args=(--no-deduplicate)
fi

short_generation_args=()
if [[ "${ALLOW_SHORT_GENERATION:-false}" == "true" ]]; then
  short_generation_args=(--allow_short_generation)
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "Augmenting train split only:"
echo "  input:  ${DATASET_DIR}"
echo "  output: ${OUTPUT_DIR}"
echo "  model:  ${MODEL_NAME_OR_PATH}"
echo "  downsample: ratio=${DOWNSAMPLE_RATIO}, seed=${SEED}, group_key=${DOWNSAMPLE_GROUP_KEY}"
echo
echo "After this, train on OUTPUT_DIR with DOWNSAMPLE_RATIO=1.0 to avoid double-downsampling."

python \
  scripts/augment_ranking_dataset_with_generations.py \
  --dataset_name "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --num_new_responses "${NUM_NEW_RESPONSES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --prompt_format "${PROMPT_FORMAT}" \
  --splits train validation test \
  --augment_splits train \
  --dataset_downsample_ratio "${DOWNSAMPLE_RATIO}" \
  --dataset_downsample_seed "${SEED}" \
  --dataset_downsample_group_key "${DOWNSAMPLE_GROUP_KEY}" \
  --dataset_downsample_splits train \
  "${overwrite_args[@]}" \
  "${max_rows_args[@]}" \
  "${deduplicate_args[@]}" \
  "${short_generation_args[@]}" \
  "$@"
