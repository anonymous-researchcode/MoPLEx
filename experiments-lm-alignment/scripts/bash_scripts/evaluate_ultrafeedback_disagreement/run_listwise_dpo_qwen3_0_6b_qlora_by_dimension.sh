#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_listwise_qlora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
DIMENSIONS="${DIMENSIONS:-instruction_following helpfulness honesty truthfulness}"
SEEDS="${SEEDS:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"
LISTWISE_NUM_RESPONSES="${LISTWISE_NUM_RESPONSES:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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

  for seed in ${SEEDS}; do
    run_name="${WANDB_NAME:-qwen3-0.6b-listwise-dpo-qlora-ultrafeedback-disagreement}-${dim_tag}-s${seed}"
    output_dir="${OUTPUT_DIR:-outputs/ultrafeedback-disagreement/listwise-dpo-by-dimension/qwen3-0.6b}-${dim_tag}-s${seed}"

    echo "Launching ListDPO for preference_dimension=${dimension}, seed=${seed}"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
      --config_file recipes/accelerate_configs/single.yaml \
      --num_processes="${NUM_PROCESSES:-1}" \
      scripts/dpo.py \
      --config "${CONFIG_PATH}" \
      --dataset_name "${DATASET_DIR}" \
      --dataset_format listwise \
      --dataset_train_split train \
      --dataset_test_split validation \
      --preference_dimensions "${dimension}" \
      --listwise_num_responses "${LISTWISE_NUM_RESPONSES}" \
      --output_dir "${output_dir}" \
      --run_name "${run_name}" \
      --report_to wandb \
      --seed "${seed}" \
      --max_steps "${MAX_STEPS}" \
      "$@"
  done
done
