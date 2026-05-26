#!/usr/bin/env bash

PERSONA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSONA_BASH_SCRIPTS_DIR="$(cd "${PERSONA_SCRIPT_DIR}/.." && pwd)"
PERSONA_SCRIPTS_DIR="$(cd "${PERSONA_BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PERSONA_SCRIPTS_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/persona_datasets}"
PROMPT_MODE="${PROMPT_MODE:-instruction_only}"
NUM_PERSONAS="${NUM_PERSONAS:-10}"
NEGATIVE_RESPONSE_SOURCE="${NEGATIVE_RESPONSE_SOURCE:-other_persona}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"

if [[ -z "${EVAL_NUM_RESPONSES_PER_PROMPT+x}" ]]; then
  if [[ "${NEGATIVE_RESPONSE_SOURCE}" == "other_persona" ]]; then
    EVAL_NUM_RESPONSES_PER_PROMPT="2"
  else
    EVAL_NUM_RESPONSES_PER_PROMPT=""
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${NEGATIVE_RESPONSE_SOURCE}" == "original" ]]; then
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}}"
else
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}_${NEGATIVE_RESPONSE_SOURCE}}"
fi

RUN_TAG="n${NUM_PERSONAS}-${PROMPT_MODE}-${NEGATIVE_RESPONSE_SOURCE}-s${SEED}"

persona_listwise_k_suffix() {
  local train_k="$1"
  local suffix="k${train_k}"
  if [[ -n "${EVAL_NUM_RESPONSES_PER_PROMPT}" && "${EVAL_NUM_RESPONSES_PER_PROMPT}" != "${train_k}" ]]; then
    suffix="${suffix}_evalk${EVAL_NUM_RESPONSES_PER_PROMPT}"
  fi
  echo "${suffix}"
}

persona_pairwise_dataset_dir() {
  echo "${DATA_ROOT}/${DATASET_NAME_PREFIX}_pairwise_${PROMPT_MODE}"
}

persona_listwise_dataset_dir() {
  local train_k="$1"
  echo "${DATA_ROOT}/${DATASET_NAME_PREFIX}_top1_listwise_${PROMPT_MODE}_$(persona_listwise_k_suffix "${train_k}")"
}

persona_generate_command() {
  local train_k="$1"
  local command_text
  command_text="NUM_PERSONAS=${NUM_PERSONAS} PROMPT_MODE=${PROMPT_MODE} NEGATIVE_RESPONSE_SOURCE=${NEGATIVE_RESPONSE_SOURCE} TRAIN_NUM_RESPONSES_PER_PROMPT=${train_k}"
  if [[ -n "${EVAL_NUM_RESPONSES_PER_PROMPT}" ]]; then
    command_text="${command_text} EVAL_NUM_RESPONSES_PER_PROMPT=${EVAL_NUM_RESPONSES_PER_PROMPT}"
  fi
  command_text="${command_text} bash scripts/bash_scripts/evaluate_persona/generate_persona_subset.sh"
  echo "${command_text}"
}

require_persona_dataset() {
  local dataset_dir="$1"
  local train_k="$2"
  if [[ -d "${dataset_dir}" ]]; then
    return
  fi

  echo "Missing prepared PERSONA dataset: ${dataset_dir}" >&2
  echo "Generate it first with:" >&2
  echo "  $(persona_generate_command "${train_k}")" >&2
  exit 1
}

run_persona_dpo_command() {
  local config_path="$1"
  local dataset_name="$2"
  local run_name="$3"
  local output_dir="$4"
  shift 4

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_CONFIG:-recipes/accelerate_configs/single.yaml}" \
    --num_processes="${NUM_PROCESSES:-1}" \
    scripts/dpo.py \
    --config "${config_path}" \
    --dataset_name "${dataset_name}" \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --seed "${SEED}" \
    --max_steps "${MAX_STEPS}" \
    "$@"
}
