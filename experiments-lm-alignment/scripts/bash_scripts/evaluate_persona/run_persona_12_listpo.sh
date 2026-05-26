#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/persona_common.sh"

NUM_PERSONAS=12
LISTWISE_K="${LISTWISE_K:-2}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/persona_datasets/persona_12_other_persona_top1_listwise_instruction_only_k${LISTWISE_K}}"
SEEDS="${SEEDS:-42 43}"
MAX_STEPS="${MAX_STEPS:-2000}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing PERSONA listwise dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

for seed in ${SEEDS}; do
  SEED="${seed}" run_persona_dpo_command \
    recipes/qwen3-1b/dpo/persona/config_top1_listwise_qlora.yaml \
    "${DATASET_DIR}" \
    "qwen3-0.6b-persona12-listpo-k${LISTWISE_K}-s${seed}" \
    "outputs/persona/persona12-listpo-k${LISTWISE_K}-s${seed}" \
    --listwise true \
    --listwise_num_responses "${LISTWISE_K}" \
    --listwise_min_responses 2 \
    --run_ranking_eval true \
    --ranking_eval_during_training true \
    "$@"
done
