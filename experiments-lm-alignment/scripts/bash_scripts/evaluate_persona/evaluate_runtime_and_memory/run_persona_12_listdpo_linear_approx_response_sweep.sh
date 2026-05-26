#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATE_PERSONA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASH_SCRIPTS_DIR="$(cd "${EVALUATE_PERSONA_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/persona/config_top1_listwise_qlora.yaml}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/persona_datasets}"
NUM_PERSONAS="${NUM_PERSONAS:-12}"
PROMPT_MODE="${PROMPT_MODE:-instruction_only}"
NEGATIVE_RESPONSE_SOURCE="${NEGATIVE_RESPONSE_SOURCE:-other_persona}"
EVAL_NUM_RESPONSES_PER_PROMPT="${EVAL_NUM_RESPONSES_PER_PROMPT:-2}"

TRAIN_RESPONSE_COUNTS="${TRAIN_RESPONSE_COUNTS:-2 8 14 20 26 32}"
ANCHORS="${ANCHORS:-0 2 4}"
SEEDS="${SEEDS:-42 43}"
MAX_STEPS="${MAX_STEPS:-2000}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-5e-6}}"
DATASET_DOWNSAMPLE_RATIO="${DATASET_DOWNSAMPLE_RATIO:-${DOWNSAMPLE_RATIO:-1.0}}"
DATASET_DOWNSAMPLE_GROUP_KEY="${DATASET_DOWNSAMPLE_GROUP_KEY:-source_index}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-ranking_validation/ranking/top1_acc}"

# Linear approximation uses input-embedding gradients. Disabling checkpointing
# makes runtime/memory comparisons less noisy and avoids recomputation issues.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
LINEAR_APPROX_REF_MODE="${LINEAR_APPROX_REF_MODE:-exact_score}"
PREPARE_MISSING="${PREPARE_MISSING:-false}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/persona/runtime-memory/listdpo-linear-approx}"
BASE_WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-persona12-listdpo-linear-approx-runtime-memory}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"
EXTRA_ARGS=("$@")

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${NEGATIVE_RESPONSE_SOURCE}" == "original" ]]; then
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}}"
else
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}_${NEGATIVE_RESPONSE_SOURCE}}"
fi

if [[ -n "${DIMENSIONS:-}" ]]; then
  read -r -a dimension_args <<< "${DIMENSIONS}"
else
  dimension_args=()
  for ((persona_idx = 0; persona_idx < NUM_PERSONAS; persona_idx++)); do
    dimension_args+=("persona_$(printf "%04d" "${persona_idx}")")
  done
  DIMENSIONS="${dimension_args[*]}"
fi

listwise_suffix_for_k() {
  local train_k="$1"
  local suffix="k${train_k}"
  if [[ -n "${EVAL_NUM_RESPONSES_PER_PROMPT}" && "${EVAL_NUM_RESPONSES_PER_PROMPT}" != "${train_k}" ]]; then
    suffix="${suffix}_evalk${EVAL_NUM_RESPONSES_PER_PROMPT}"
  fi
  echo "${suffix}"
}

listwise_dataset_dir_for_k() {
  local train_k="$1"
  echo "${DATA_ROOT}/${DATASET_NAME_PREFIX}_top1_listwise_${PROMPT_MODE}_$(listwise_suffix_for_k "${train_k}")"
}

prepare_dataset_for_k() {
  local train_k="$1"
  NUM_PERSONAS="${NUM_PERSONAS}" \
  PROMPT_MODE="${PROMPT_MODE}" \
  NEGATIVE_RESPONSE_SOURCE="${NEGATIVE_RESPONSE_SOURCE}" \
  TRAIN_NUM_RESPONSES_PER_PROMPT="${train_k}" \
  EVAL_NUM_RESPONSES_PER_PROMPT="${EVAL_NUM_RESPONSES_PER_PROMPT}" \
  bash "${EVALUATE_PERSONA_DIR}/generate_persona_subset.sh"
}

require_dataset_for_k() {
  local train_k="$1"
  local dataset_dir="$2"
  if [[ -d "${dataset_dir}" ]]; then
    return
  fi
  if [[ "${PREPARE_MISSING}" == "true" ]]; then
    prepare_dataset_for_k "${train_k}"
    return
  fi
  echo "Missing PERSONA top-1 listwise dataset: ${dataset_dir}" >&2
  echo "Generate it with:" >&2
  echo "  NUM_PERSONAS=${NUM_PERSONAS} TRAIN_NUM_RESPONSES_PER_PROMPT=${train_k} EVAL_NUM_RESPONSES_PER_PROMPT=${EVAL_NUM_RESPONSES_PER_PROMPT} bash scripts/bash_scripts/evaluate_persona/generate_persona_subset.sh" >&2
  echo "Or rerun this script with PREPARE_MISSING=true." >&2
  exit 1
}

run_one() {
  local train_k="$1"
  local dataset_dir="$2"
  local anchor="$3"
  local seed="$4"
  local learning_rate="$5"

  local dataset_tag
  local lr_tag
  local ratio_tag
  local run_name
  local output_dir

  dataset_tag="$(basename "${dataset_dir}")"
  lr_tag="${learning_rate//./p}"
  lr_tag="${lr_tag//-/m}"
  ratio_tag="${DATASET_DOWNSAMPLE_RATIO//./p}"

  run_name="${BASE_WANDB_NAME}-m${train_k}-a${anchor}-ds${ratio_tag}-lr${lr_tag}-s${seed}"
  output_dir="${OUTPUT_ROOT}/${dataset_tag}/m${train_k}-a${anchor}-ds${ratio_tag}-lr${lr_tag}-s${seed}"

  approx_args=()
  approx_label="exact"
  if [[ "${anchor}" -gt 0 ]]; then
    approx_args=(
      --use_linear_reward_approx true
      --linear_approx_num_anchors "${anchor}"
      --linear_approx_exact_eval true
      --linear_approx_ref_mode "${LINEAR_APPROX_REF_MODE}"
    )
    approx_label="linear_approx"
  fi

  echo "Launching PERSONA-12 ListDPO runtime/memory run: dataset=${dataset_dir}; m=${train_k}; a=${anchor}; mode=${approx_label}; ranked_prefix_length=1; dims=${DIMENSIONS}; seed=${seed}; lr=${learning_rate}; downsample=${DATASET_DOWNSAMPLE_RATIO}; ref_mode=${LINEAR_APPROX_REF_MODE}"

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_CONFIG:-recipes/accelerate_configs/single.yaml}" \
    --num_processes="${NUM_PROCESSES:-1}" \
    scripts/dpo.py \
    --config "${CONFIG_PATH}" \
    --dataset_name "${dataset_dir}" \
    --dataset_format listwise \
    --dataset_train_split train \
    --dataset_test_split validation \
    --preference_dimensions "${dimension_args[@]}" \
    --listwise true \
    --listwise_num_responses "${train_k}" \
    --listwise_min_responses 2 \
    --run_ranking_eval true \
    --ranking_eval_during_training false \
    --dataset_downsample_ratio "${DATASET_DOWNSAMPLE_RATIO}" \
    --dataset_downsample_seed "${seed}" \
    --dataset_downsample_group_key "${DATASET_DOWNSAMPLE_GROUP_KEY}" \
    "${approx_args[@]}" \
    --learning_rate "${learning_rate}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --eval_strategy "${EVAL_STRATEGY:-no}" \
    --load_best_model_at_end false \
    --metric_for_best_model "${METRIC_FOR_BEST_MODEL}" \
    --eval_steps "${EVAL_STEPS:-500}" \
    --save_steps "${SAVE_STEPS:-500}" \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --report_to "${REPORT_TO:-wandb}" \
    --seed "${seed}" \
    "${EXTRA_ARGS[@]}"
}

for learning_rate in ${LEARNING_RATES}; do
for anchor in ${ANCHORS}; do
for train_k in ${TRAIN_RESPONSE_COUNTS}; do
for seed in ${SEEDS}; do
  dataset_dir="$(listwise_dataset_dir_for_k "${train_k}")"
  require_dataset_for_k "${train_k}" "${dataset_dir}"
  run_one "${train_k}" "${dataset_dir}" "${anchor}" "${seed}" "${learning_rate}"
done
done
done
done
