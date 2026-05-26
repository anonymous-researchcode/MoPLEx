#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
MICRO_ROOT="${MICRO_ROOT:-${WORKSPACE_ROOT}/MiCRo}"

lr="${LR:-2e-3}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
bs="${BS:-4}"
eval_bs="${EVAL_BS:-4}"
num_heads="${NUM_HEADS:-12}"
loss_type="${LOSS_TYPE:-mixture_BT}"
num_train_epochs="${NUM_TRAIN_EPOCHS:-5}"
max_steps="${MAX_STEPS:-2000}"
downsample_rate="${DOWNSAMPLE_RATE:-1.0}"
seeds="${SEEDS:-42 43}"

orthogonal_loss_weight="${ORTHOGONAL_LOSS_WEIGHT:-0}"
norm_loss_weight="${NORM_LOSS_WEIGHT:-0}"
corr_loss_weight="${CORR_LOSS_WEIGHT:-0}"
load_balance_loss_weight="${LOAD_BALANCE_LOSS_WEIGHT:-0.5}"
use_router="${USE_ROUTER:-True}"

data_path="${DATA_PATH:-${REPO_ROOT}/data/persona_datasets/persona_12_other_persona_top1_listwise_instruction_only_k2}"
log_dir="${LOG_DIR:-${REPO_ROOT}/logs/persona_12}"

if [[ ! -d "${data_path}" ]]; then
  echo "Missing PERSONA listwise dataset directory: ${data_path}" >&2
  exit 1
fi

mkdir -p "${log_dir}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${MICRO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for seed in ${seeds}; do
  run_name="persona12_micro_cluster_${loss_type}_heads${num_heads}_steps${max_steps}_seed${seed}"
  log_file="${log_dir}/${run_name}.log"
  rm -f "${log_file}"

  accelerate launch \
    --config_file "${ACCELERATE_CONFIG:-recipes/accelerate_configs/single.yaml}" \
    --num_processes="${NUM_PROCESSES:-1}" \
    src/alignment/micro.py \
      --learning_rate="${lr}" \
      --loss_type="${loss_type}" \
      --num_heads="${num_heads}" \
      --wandb_name="${run_name}" \
      --data_path="${data_path}" \
      --per_device_train_batch_size="${bs}" \
      --per_device_eval_batch_size="${eval_bs}" \
      --num_train_epochs="${num_train_epochs}" \
      --max_steps="${max_steps}" \
      --gradient_accumulation_steps="${gradient_accumulation_steps}" \
      --orthogonal_loss_weight="${orthogonal_loss_weight}" \
      --norm_loss_weight="${norm_loss_weight}" \
      --corr_loss_weight="${corr_loss_weight}" \
      --use_router="${use_router}" \
      --load_balance_loss_weight="${load_balance_loss_weight}" \
      --base_model="${BASE_MODEL:-Qwen/Qwen3-0.6B}" \
      --downsample_rate="${downsample_rate}" \
      --manual_seed="${seed}" \
      "$@" \
      | tee -a "${log_file}"
done
