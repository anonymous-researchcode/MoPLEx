#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
SOURCE_DATASET="${SOURCE_DATASET:-SynthLabsAI/PERSONA}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/persona_datasets}"
PROMPT_MODE="${PROMPT_MODE:-instruction_only}"
NUM_PERSONAS="${NUM_PERSONAS:-12}"
TRAIN_NUM_RESPONSES_PER_PROMPT="${TRAIN_NUM_RESPONSES_PER_PROMPT:-8 14 20 26 32}"
EVAL_NUM_RESPONSES_PER_PROMPT="${EVAL_NUM_RESPONSES_PER_PROMPT:-2}"
HARD_NEGATIVE_PERSONA_POOL="${HARD_NEGATIVE_PERSONA_POOL:-all}"
NEGATIVE_POOL="${NEGATIVE_POOL:-same_split}"
NEGATIVE_RESPONSE_SOURCE="${NEGATIVE_RESPONSE_SOURCE:-other_persona}"
SPLIT_MODE="${SPLIT_MODE:-prompt}"
SEED="${SEED:-42}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${NEGATIVE_RESPONSE_SOURCE}" == "original" ]]; then
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}}"
else
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}_${NEGATIVE_RESPONSE_SOURCE}}"
fi

LISTWISE_K_SUFFIX="k${TRAIN_NUM_RESPONSES_PER_PROMPT}"
eval_num_responses_args=()
if [[ -n "${EVAL_NUM_RESPONSES_PER_PROMPT}" ]]; then
  eval_num_responses_args=(--eval_num_responses_per_prompt "${EVAL_NUM_RESPONSES_PER_PROMPT}")
  if [[ "${EVAL_NUM_RESPONSES_PER_PROMPT}" != "${TRAIN_NUM_RESPONSES_PER_PROMPT}" ]]; then
    LISTWISE_K_SUFFIX="${LISTWISE_K_SUFFIX}_evalk${EVAL_NUM_RESPONSES_PER_PROMPT}"
  fi
fi

PAIRWISE_DATASET_DIR="${DATA_ROOT}/${DATASET_NAME_PREFIX}_pairwise_${PROMPT_MODE}"
LISTWISE_DATASET_DIR="${DATA_ROOT}/${DATASET_NAME_PREFIX}_top1_listwise_${PROMPT_MODE}_${LISTWISE_K_SUFFIX}"

overwrite_args=()
if [[ "${FORCE_PREPARE}" == "true" || -d "${PAIRWISE_DATASET_DIR}" || -d "${LISTWISE_DATASET_DIR}" ]]; then
  overwrite_args=(--overwrite)
fi

if [[ "${FORCE_PREPARE}" != "true" && -d "${PAIRWISE_DATASET_DIR}" && -d "${LISTWISE_DATASET_DIR}" ]]; then
  echo "PERSONA datasets already exist:"
  echo "  pairwise: ${PAIRWISE_DATASET_DIR}"
  echo "  listwise: ${LISTWISE_DATASET_DIR}"
  echo "Set FORCE_PREPARE=true to regenerate them."
  exit 0
fi

echo "Preparing reusable PERSONA subset datasets"
echo "  num_personas: ${NUM_PERSONAS}"
echo "  prompt_mode: ${PROMPT_MODE}"
echo "  negative_response_source: ${NEGATIVE_RESPONSE_SOURCE}"
echo "  train_num_responses: ${TRAIN_NUM_RESPONSES_PER_PROMPT}"
echo "  eval_num_responses: ${EVAL_NUM_RESPONSES_PER_PROMPT:-same as train}"
echo "  pairwise: ${PAIRWISE_DATASET_DIR}"
echo "  listwise: ${LISTWISE_DATASET_DIR}"

for train_num_responses_per_prompt in ${TRAIN_NUM_RESPONSES_PER_PROMPT}; do
"${PYTHON}" scripts/prepare_persona_baselines_dataset.py \
  --dataset_name "${SOURCE_DATASET}" \
  --output_dir "${DATA_ROOT}" \
  --output_name_prefix "${DATASET_NAME_PREFIX}" \
  --num_personas "${NUM_PERSONAS}" \
  --num_responses_per_prompt "${train_num_responses_per_prompt}" \
  --prompt_mode "${PROMPT_MODE}" \
  --hard_negative_persona_pool "${HARD_NEGATIVE_PERSONA_POOL}" \
  --negative_pool "${NEGATIVE_POOL}" \
  --negative_response_source "${NEGATIVE_RESPONSE_SOURCE}" \
  --split_mode "${SPLIT_MODE}" \
  --persona_subset_seed "${SEED}" \
  --split_seed "${SEED}" \
  --hard_negative_seed "${SEED}" \
  "${eval_num_responses_args[@]}" \
  "${overwrite_args[@]}"
done

echo "Prepared PERSONA datasets:"
echo "  pairwise: ${PAIRWISE_DATASET_DIR}"
echo "  listwise: ${LISTWISE_DATASET_DIR}"
