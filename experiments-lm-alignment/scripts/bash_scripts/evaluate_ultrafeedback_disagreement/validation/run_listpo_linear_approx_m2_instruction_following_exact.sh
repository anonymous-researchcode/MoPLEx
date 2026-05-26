#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_listwise_qlora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
DIMENSION="${DIMENSION:-instruction_following}"
SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"
LISTWISE_NUM_RESPONSES="${LISTWISE_NUM_RESPONSES:-2}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-0.25}"
DOWNSAMPLE_GROUP_KEY="${DOWNSAMPLE_GROUP_KEY:-source_index}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-ranking_validation/ranking/pairwise_acc}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-anonymous}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"
dataset_downsample_seed="42"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing disagreement dataset directory: ${DATASET_DIR}" >&2
  echo "Run scripts/bash_scripts/generate_disagreement_ultrafeedback_dataset.sh first, or set DATASET_DIR." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

dim_tag="${DIMENSION//[^a-zA-Z0-9]/_}"
ratio_tag="${DOWNSAMPLE_RATIO//./p}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-outputs/ultrafeedback-disagreement/listpo-m2-linear-approx/qwen3-0.6b}"
BASE_WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-listpo-m2-linear-approx}"

for seed in ${SEEDS}; do
  for mode in exact; do
    approx_args=()
    mode_tag="${mode}"
    if [[ "${mode}" == "approx_a1" ]]; then
      approx_args=(
        --use_linear_reward_approx true
        --linear_approx_num_anchors 1
        --linear_approx_exact_eval true
      )
      mode_tag="approx-a1"
    fi

    run_name="${BASE_WANDB_NAME}-${mode_tag}-${dim_tag}-ds${ratio_tag}-s${seed}"
    output_dir="${BASE_OUTPUT_DIR}/${mode_tag}-${dim_tag}-ds${ratio_tag}-s${seed}"

    echo "Launching ListPO m=${LISTWISE_NUM_RESPONSES}: mode=${mode}; dimension=${DIMENSION}; downsample=${DOWNSAMPLE_RATIO}; seed=${seed}"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
      --config_file recipes/accelerate_configs/single.yaml \
      --num_processes="${NUM_PROCESSES:-1}" \
      scripts/dpo.py \
      --config "${CONFIG_PATH}" \
      --dataset_name "${DATASET_DIR}" \
      --dataset_format listwise \
      --dataset_train_split train \
      --dataset_test_split validation \
      --preference_dimensions "${DIMENSION}" \
      --listwise true \
      --listwise_num_responses "${LISTWISE_NUM_RESPONSES}" \
      --listwise_min_responses 2 \
      --run_ranking_eval true \
      --ranking_eval_during_training true \
      --dataset_downsample_ratio "${DOWNSAMPLE_RATIO}" \
      --dataset_downsample_seed "${dataset_downsample_seed}" \
      --dataset_downsample_group_key "${DOWNSAMPLE_GROUP_KEY}" \
      --metric_for_best_model "${METRIC_FOR_BEST_MODEL}" \
      --output_dir "${output_dir}" \
      --run_name "${run_name}" \
      --report_to "${REPORT_TO:-wandb}" \
      --seed "${seed}" \
      --max_steps "${MAX_STEPS}" \
      "${approx_args[@]}" \
      "$@"
  done
done
