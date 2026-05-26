#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_mixture_qlora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement_train_augmented_Qwen3_0p6B_ds0p25_k8_s42}"
BASE_STATS_PATH="${BASE_STATS_PATH:-${REPO_ROOT}/data/ultrafeedback_disagreement/stats.json}"
SEEDS="${SEEDS:-42}"
RANKING_SIZES="${RANKING_SIZES:-4}"
GENERATED_RANKING_SIZES="${GENERATED_RANKING_SIZES:-4 2}"
MAX_STEPS="${MAX_STEPS:-2000}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-2e-6}}"
EM_TEMPERATURES="${EM_TEMPERATURES:-${EM_TEMPERATURE:-1.0}}"
M_STEP_UPDATES="${M_STEP_UPDATES:-3 1}"
MIXTURE_TRAINING_MODE="${MIXTURE_TRAINING_MODE:-em_only}"
MIXTURE_REWARD_BACKEND="${MIXTURE_REWARD_BACKEND:-lora}"
# LoRA mixture switches active adapters inside the loss forward, which is not
# compatible with checkpoint recomputation.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-1.0}"
DOWNSAMPLE_GROUP_KEY="${DOWNSAMPLE_GROUP_KEY:-source_index}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-mixture_posterior/pairwise_acc}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing augmented disagreement dataset directory: ${DATASET_DIR}" >&2
  echo "Generate it first, or set DATASET_DIR to an existing augmented DatasetDict." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DIMENSIONS="${DIMENSIONS:-}"
if [[ -z "${DIMENSIONS}" ]]; then
  if [[ -f "${BASE_STATS_PATH}" ]]; then
    DIMENSIONS="$(
      /home/ldy/miniconda3/envs/alignment/bin/python -c \
        'import json, sys; stats=json.load(open(sys.argv[1])); print(" ".join(stats.get("dimensions", [])))' \
        "${BASE_STATS_PATH}"
    )"
  elif [[ -f "${DATASET_DIR}/stats.json" ]]; then
    DIMENSIONS="$(
      /home/ldy/miniconda3/envs/alignment/bin/python -c \
        'import json, sys; stats=json.load(open(sys.argv[1])); print(" ".join(stats.get("dimensions", [])))' \
        "${DATASET_DIR}/stats.json"
    )"
  fi
fi
DIMENSIONS="${DIMENSIONS:-instruction_following honesty truthfulness helpfulness}"
read -r -a dimension_args <<< "${DIMENSIONS}"
NUM_CLUSTERS="${NUM_CLUSTERS:-${#dimension_args[@]}}"

downsample_split_args=()
if [[ -n "${DOWNSAMPLE_SPLITS:-}" ]]; then
  read -r -a downsample_split_args <<< "${DOWNSAMPLE_SPLITS}"
  downsample_split_args=(--dataset_downsample_splits "${downsample_split_args[@]}")
fi

dataset_tag="$(basename "${DATASET_DIR}")"
ratio_tag="${DOWNSAMPLE_RATIO//./p}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-outputs/ultrafeedback-disagreement/mixture-pl-lora-augmented/qwen3-0.6b-${dataset_tag}}"
BASE_WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-mixture-pl-lora-augmented-${dataset_tag}}"

for seed in ${SEEDS}; do
for learning_rate in ${LEARNING_RATES}; do
for temperature in ${EM_TEMPERATURES}; do
for generated_ranking_size in ${GENERATED_RANKING_SIZES}; do
for m_step_updates in ${M_STEP_UPDATES}; do
for ranking_size in ${RANKING_SIZES}; do
  lr_tag="${learning_rate//./p}"
  lr_tag="${lr_tag//-/m}"
  temp_tag="${temperature//./p}"
  run_name="${BASE_WANDB_NAME}-m${ranking_size}-mp${generated_ranking_size}-ms${m_step_updates}-ds${ratio_tag}-temp${temp_tag}-lr${lr_tag}-s${seed}"
  output_dir="${BASE_OUTPUT_DIR}/m${ranking_size}-mp${generated_ranking_size}-ms${m_step_updates}-ds${ratio_tag}-temp${temp_tag}-lr${lr_tag}-s${seed}"

  echo "Launching augmented mixture PL (${MIXTURE_REWARD_BACKEND} backend): dataset=${DATASET_DIR}; dims=${DIMENSIONS}; clusters=${NUM_CLUSTERS}; m=${ranking_size}; m_prime=${generated_ranking_size}; m_step_updates=${m_step_updates}; seed=${seed}; downsample=${DOWNSAMPLE_RATIO}; gradient_checkpointing=${GRADIENT_CHECKPOINTING}"

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single.yaml \
    --num_processes="${NUM_PROCESSES:-1}" \
    scripts/dpo.py \
    --config "${CONFIG_PATH}" \
    --dataset_name "${DATASET_DIR}" \
    --dataset_format listwise \
    --dataset_train_split train \
    --dataset_test_split validation \
    --preference_dimensions "${dimension_args[@]}" \
    --listwise true \
    --listwise_num_responses "${ranking_size}" \
    --listwise_num_generated_responses "${generated_ranking_size}" \
    --listwise_min_responses 2 \
    --run_ranking_eval true \
    --ranking_eval_during_training true \
    --dataset_downsample_ratio "${DOWNSAMPLE_RATIO}" \
    --dataset_downsample_seed "${seed}" \
    --dataset_downsample_group_key "${DOWNSAMPLE_GROUP_KEY}" \
    "${downsample_split_args[@]}" \
    --use_mixture true \
    --mixture_objective pl \
    --num_clusters "${NUM_CLUSTERS}" \
    --mixture_training_mode "${MIXTURE_TRAINING_MODE}" \
    --mixture_reward_backend "${MIXTURE_REWARD_BACKEND}" \
    --mixture_nll_weight "${MIXTURE_NLL_WEIGHT:-0.1}" \
    --em_temperature "${temperature}" \
    --m_step_updates "${m_step_updates}" \
    --use_contextual_router "${USE_CONTEXTUAL_ROUTER:-false}" \
    --use_closed_form_router_prior_update "${USE_CLOSED_FORM_ROUTER_PRIOR_UPDATE:-false}" \
    --router_hidden_size "${ROUTER_HIDDEN_SIZE:-256}" \
    --log_cluster_metrics true \
    --learning_rate "${learning_rate}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --eval_steps "${EVAL_STEPS:-200}" \
    --save_steps "${SAVE_STEPS:-200}" \
    --metric_for_best_model "${METRIC_FOR_BEST_MODEL}" \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --report_to "${REPORT_TO:-wandb}" \
    --seed "${seed}" \
    "$@"
done
done
done
done
done
done
