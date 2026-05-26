#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_mixture_qlora.yaml}"
ORIGINAL_DATASET_DIR="${ORIGINAL_DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
AUGMENTED_DATASET_DIR="${AUGMENTED_DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement_train_augmented_Qwen3_0p6B_ds0p25_k8_s42}"
BASE_STATS_PATH="${BASE_STATS_PATH:-${ORIGINAL_DATASET_DIR}/stats.json}"

DATASET_MODES="${DATASET_MODES:-augmented}"
ANCHORS="${ANCHORS:-2 4}"
SEEDS="${SEEDS:-42}"
RANKING_SIZES="${RANKING_SIZES:-4}"
GENERATED_RANKING_SIZES="${GENERATED_RANKING_SIZES:-4 2}"
MAX_STEPS="${MAX_STEPS:-2000}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-2e-6}}"
EM_TEMPERATURES="${EM_TEMPERATURES:-${EM_TEMPERATURE:-1.0}}"
M_STEP_UPDATES="${M_STEP_UPDATES:-3 1}"
MIXTURE_TRAINING_MODE="${MIXTURE_TRAINING_MODE:-em_only}"
MIXTURE_REWARD_BACKEND="${MIXTURE_REWARD_BACKEND:-lora}"
ORIGINAL_DOWNSAMPLE_RATIO="${ORIGINAL_DOWNSAMPLE_RATIO:-${DOWNSAMPLE_RATIO:-0.25}}"
AUGMENTED_DOWNSAMPLE_RATIO="${AUGMENTED_DOWNSAMPLE_RATIO:-${DOWNSAMPLE_RATIO:-1.0}}"
DOWNSAMPLE_GROUP_KEY="${DOWNSAMPLE_GROUP_KEY:-source_index}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-mixture_posterior/pairwise_acc}"

# The approximation path calls autograd.grad for input-embedding gradients.
# This is not compatible with checkpoint recomputation while LoRA adapters are
# switched inside the mixture loss.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
LINEAR_APPROX_REF_MODE="${LINEAR_APPROX_REF_MODE:-exact_score}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"
EXTRA_ARGS=("$@")

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
  elif [[ -f "${ORIGINAL_DATASET_DIR}/stats.json" ]]; then
    DIMENSIONS="$(
      /home/ldy/miniconda3/envs/alignment/bin/python -c \
        'import json, sys; stats=json.load(open(sys.argv[1])); print(" ".join(stats.get("dimensions", [])))' \
        "${ORIGINAL_DATASET_DIR}/stats.json"
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

BASE_OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_DIR:-outputs/ultrafeedback-disagreement/mixture-pl-lora-linear-approx}}"
BASE_WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-mixture-pl-lora-linear-approx}"

run_one() {
  local dataset_mode="$1"
  local dataset_dir="$2"
  local downsample_ratio="$3"
  local anchor="$4"
  local seed="$5"
  local learning_rate="$6"
  local temperature="$7"
  local m_step_updates="$8"
  local ranking_size="$9"
  local generated_ranking_size="${10}"

  local dataset_tag
  local ratio_tag
  local lr_tag
  local temp_tag
  local run_name
  local output_dir
  local generated_args=()
  local generated_tag=""
  local generated_echo=""

  dataset_tag="$(basename "${dataset_dir}")"
  ratio_tag="${downsample_ratio//./p}"
  lr_tag="${learning_rate//./p}"
  lr_tag="${lr_tag//-/m}"
  temp_tag="${temperature//./p}"

  if [[ "${dataset_mode}" == "augmented" ]]; then
    generated_args=(--listwise_num_generated_responses "${generated_ranking_size}")
    generated_tag="-mp${generated_ranking_size}"
    generated_echo="; m_prime=${generated_ranking_size}"
  fi

  run_name="${BASE_WANDB_NAME}-${dataset_mode}-approx-a${anchor}-m${ranking_size}${generated_tag}-ms${m_step_updates}-ds${ratio_tag}-temp${temp_tag}-lr${lr_tag}-s${seed}"
  output_dir="${BASE_OUTPUT_ROOT}/${dataset_mode}/qwen3-0.6b-${dataset_tag}/approx-a${anchor}-m${ranking_size}${generated_tag}-ms${m_step_updates}-ds${ratio_tag}-temp${temp_tag}-lr${lr_tag}-s${seed}"

  echo "Launching ${dataset_mode} approximate mixture PL (${MIXTURE_REWARD_BACKEND} backend): dataset=${dataset_dir}; dims=${DIMENSIONS}; clusters=${NUM_CLUSTERS}; a=${anchor}; m=${ranking_size}${generated_echo}; m_step_updates=${m_step_updates}; seed=${seed}; downsample=${downsample_ratio}; ref_mode=${LINEAR_APPROX_REF_MODE}; gradient_checkpointing=${GRADIENT_CHECKPOINTING}"

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single.yaml \
    --num_processes="${NUM_PROCESSES:-1}" \
    scripts/dpo.py \
    --config "${CONFIG_PATH}" \
    --dataset_name "${dataset_dir}" \
    --dataset_format listwise \
    --dataset_train_split train \
    --dataset_test_split validation \
    --preference_dimensions "${dimension_args[@]}" \
    --listwise true \
    --listwise_num_responses "${ranking_size}" \
    "${generated_args[@]}" \
    --listwise_min_responses 2 \
    --run_ranking_eval true \
    --ranking_eval_during_training true \
    --dataset_downsample_ratio "${downsample_ratio}" \
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
    --use_linear_reward_approx true \
    --linear_approx_num_anchors "${anchor}" \
    --linear_approx_exact_eval true \
    --linear_approx_ref_mode "${LINEAR_APPROX_REF_MODE}" \
    --learning_rate "${learning_rate}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --eval_steps "${EVAL_STEPS:-500}" \
    --save_steps "${SAVE_STEPS:-500}" \
    --metric_for_best_model "${METRIC_FOR_BEST_MODEL}" \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --report_to "${REPORT_TO:-wandb}" \
    --seed "${seed}" \
    "${EXTRA_ARGS[@]}"
}

for dataset_mode in ${DATASET_MODES}; do
  case "${dataset_mode}" in
    original)
      dataset_dir="${ORIGINAL_DATASET_DIR}"
      downsample_ratio="${ORIGINAL_DOWNSAMPLE_RATIO}"
      generated_sizes="none"
      ;;
    augmented)
      dataset_dir="${AUGMENTED_DATASET_DIR}"
      downsample_ratio="${AUGMENTED_DOWNSAMPLE_RATIO}"
      generated_sizes="${GENERATED_RANKING_SIZES}"
      ;;
    *)
      echo "Unknown DATASET_MODES entry: ${dataset_mode}. Expected 'original' or 'augmented'." >&2
      exit 1
      ;;
  esac

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Missing ${dataset_mode} dataset directory: ${dataset_dir}" >&2
    exit 1
  fi

  for seed in ${SEEDS}; do
  for learning_rate in ${LEARNING_RATES}; do
  for temperature in ${EM_TEMPERATURES}; do
  for m_step_updates in ${M_STEP_UPDATES}; do
  for ranking_size in ${RANKING_SIZES}; do
  for anchor in ${ANCHORS}; do
    if [[ "${dataset_mode}" == "augmented" ]]; then
      for generated_ranking_size in ${generated_sizes}; do
        # skip anchor==2 and generated_ranking_size==4
        if [[ "${anchor}" -eq 2 && "${ranking_size}" -eq 4 
          && "${generated_ranking_size}" -eq 4  ]]; then
          echo "Skipping anchor=${anchor} with ranking_size=${ranking_size} and generated_ranking_size=4." >&2
          continue
        fi
        run_one "${dataset_mode}" "${dataset_dir}" "${downsample_ratio}" "${anchor}" "${seed}" "${learning_rate}" "${temperature}" "${m_step_updates}" "${ranking_size}" "${generated_ranking_size}"
      done
    else
      run_one "${dataset_mode}" "${dataset_dir}" "${downsample_ratio}" "${anchor}" "${seed}" "${learning_rate}" "${temperature}" "${m_step_updates}" "${ranking_size}" ""
    fi
  done
  done
  done
  done
  done
done
done
