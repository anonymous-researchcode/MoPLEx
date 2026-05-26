#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_dpo_qlora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
DIMENSIONS="${DIMENSIONS:-instruction_following helpfulness honesty truthfulness}"
SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-${EVAL_STEPS}}"
LISTWISE_NUM_RESPONSES="${LISTWISE_NUM_RESPONSES:-4}"
PAIRWISE_STRATEGY="${PAIRWISE_STRATEGY:-all_pairs}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-0.25}"
DATASET_DOWNSAMPLE_GROUP_KEY="${DATASET_DOWNSAMPLE_GROUP_KEY:-source_index}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-anonymous}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing disagreement dataset directory: ${DATASET_DIR}" >&2
  echo "Run scripts/bash_scripts/generate_disagreement_ultrafeedback_dataset.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

for dimension in ${DIMENSIONS}; do
  dim_tag="${dimension//[^a-zA-Z0-9]/_}"
  ratio_tag="${DOWNSAMPLE_RATIO//./p}"

  for seed in ${SEEDS}; do
    run_name="${WANDB_NAME:-qwen3-0.6b-dpo-qlora-ultrafeedback-disagreement}-${dim_tag}-ds${ratio_tag}-s${seed}"
    output_dir="${OUTPUT_DIR:-outputs/ultrafeedback-disagreement/dpo-by-dimension/qwen3-0.6b-ds${ratio_tag}}-${dim_tag}-s${seed}"

    echo "Launching pairwise DPO for preference_dimension=${dimension}, seed=${seed}, downsample=${DOWNSAMPLE_RATIO}"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
      --config_file recipes/accelerate_configs/single.yaml \
      --num_processes="${NUM_PROCESSES:-1}" \
      scripts/dpo.py \
      --config "${CONFIG_PATH}" \
      --dataset_name "${DATASET_DIR}" \
      --dataset_format pairwise \
      --dataset_train_split train \
      --dataset_test_split validation \
      --preference_dimensions "${dimension}" \
      --listwise_num_responses "${LISTWISE_NUM_RESPONSES}" \
      --pairwise_from_listwise_strategy "${PAIRWISE_STRATEGY}" \
      --dataset_downsample_ratio "${DOWNSAMPLE_RATIO}" \
      --dataset_downsample_seed "${seed}" \
      --dataset_downsample_group_key "${DATASET_DOWNSAMPLE_GROUP_KEY}" \
      --metric_for_best_model rewards/accuracies \
      --output_dir "${output_dir}" \
      --run_name "${run_name}" \
      --report_to wandb \
      --seed "${seed}" \
      --max_steps "${MAX_STEPS}" \
      --eval_steps "${EVAL_STEPS}" \
      --save_steps "${SAVE_STEPS}" \
      "$@"
  done
done
