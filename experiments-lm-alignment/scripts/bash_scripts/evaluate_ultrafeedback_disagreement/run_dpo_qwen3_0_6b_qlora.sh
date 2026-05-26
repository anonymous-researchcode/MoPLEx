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
MAX_STEPS="${MAX_STEPS:-2000}"
LISTWISE_NUM_RESPONSES="${LISTWISE_NUM_RESPONSES:-4}"
PAIRWISE_STRATEGY="${PAIRWISE_STRATEGY:-all_pairs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing disagreement dataset directory: ${DATASET_DIR}" >&2
  echo "Run scripts/bash_scripts/generate_disagreement_ultrafeedback_dataset.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

read -r -a dimension_args <<< "${DIMENSIONS}"

for seed in ${SEEDS}; do
  run_name="${WANDB_NAME:-qwen3-0.6b-dpo-qlora-ultrafeedback-disagreement-all}-s${seed}"
  output_dir="${OUTPUT_DIR:-outputs/ultrafeedback-disagreement/dpo/qwen3-0.6b-all}-s${seed}"

  echo "Launching pairwise DPO for all preference dimensions: ${DIMENSIONS}; seed=${seed}"

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single.yaml \
    --num_processes="${NUM_PROCESSES:-1}" \
    scripts/dpo.py \
    --config "${CONFIG_PATH}" \
    --dataset_name "${DATASET_DIR}" \
    --dataset_format pairwise \
    --dataset_train_split train \
    --dataset_test_split validation \
    --preference_dimensions "${dimension_args[@]}" \
    --listwise_num_responses "${LISTWISE_NUM_RESPONSES}" \
    --pairwise_from_listwise_strategy "${PAIRWISE_STRATEGY}" \
    --metric_for_best_model rewards/accuracies \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --report_to wandb \
    --seed "${seed}" \
    --max_steps "${MAX_STEPS}" \
    "$@"
done
