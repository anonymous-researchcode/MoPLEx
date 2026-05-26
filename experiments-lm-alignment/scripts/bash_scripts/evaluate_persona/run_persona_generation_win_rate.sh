#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/persona_datasets/persona_12_other_persona_top1_listwise_instruction_only_k12_evalk2}"
SPLIT="${SPLIT:-validation}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
POLICY_ADAPTER_MODE="${POLICY_ADAPTER_MODE:-single}"
MAX_EXAMPLES="${MAX_EXAMPLES:-120}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-balanced_by_dimension}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
REWARD_BETA="${REWARD_BETA:-0.01}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/persona/generation-win-rate}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing PERSONA dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

if [[ -z "${POLICY_ADAPTER_PATH:-}" ]]; then
  echo "POLICY_ADAPTER_PATH must point to a fine-tuned LoRA adapter directory." >&2
  exit 1
fi

if [[ -z "${REWARD_ADAPTER_MAP_JSON:-}" && -z "${REWARD_ADAPTERS:-}" ]]; then
  echo "Provide reward adapters with REWARD_ADAPTER_MAP_JSON or REWARD_ADAPTERS='persona_0000=/path ...'." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

dataset_tag="$(basename "${DATASET_DIR}")"
policy_tag="$(basename "${POLICY_ADAPTER_PATH}")"
output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/${dataset_tag}/${policy_tag}-s${SEED}}"

reward_args=()
if [[ -n "${REWARD_ADAPTER_MAP_JSON:-}" ]]; then
  reward_args+=(--reward_adapter_map_json "${REWARD_ADAPTER_MAP_JSON}")
fi
if [[ -n "${REWARD_ADAPTERS:-}" ]]; then
  for mapping in ${REWARD_ADAPTERS}; do
    reward_args+=(--reward_adapter "${mapping}")
  done
fi

extra_args=("$@")
flag_args=()
if [[ "${LOAD_IN_4BIT:-false}" == "true" ]]; then
  flag_args+=(--load_in_4bit)
fi
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  flag_args+=(--overwrite)
fi

echo "Evaluating generation win-rate: dataset=${DATASET_DIR}; split=${SPLIT}; policy=${POLICY_ADAPTER_PATH}; mode=${POLICY_ADAPTER_MODE}; max_examples=${MAX_EXAMPLES}; output=${output_dir}"

python scripts/evaluate_generation_win_rate.py \
  --dataset_name "${DATASET_DIR}" \
  --split "${SPLIT}" \
  --max_examples "${MAX_EXAMPLES}" \
  --sample_strategy "${SAMPLE_STRATEGY}" \
  --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
  --policy_adapter_path "${POLICY_ADAPTER_PATH}" \
  --policy_adapter_mode "${POLICY_ADAPTER_MODE}" \
  "${reward_args[@]}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --reward_beta "${REWARD_BETA}" \
  --max_length "${MAX_LENGTH}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --output_dir "${output_dir}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  "${flag_args[@]}" \
  "${extra_args[@]}"
