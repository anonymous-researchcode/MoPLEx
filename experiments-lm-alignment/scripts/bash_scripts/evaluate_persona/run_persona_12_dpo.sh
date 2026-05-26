#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/persona_common.sh"

NUM_PERSONAS=12
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/persona_datasets/persona_12_other_persona_pairwise_instruction_only}"
SEEDS="${SEEDS:-42 43}"
MAX_STEPS="${MAX_STEPS:-2000}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing PERSONA pairwise dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

for seed in ${SEEDS}; do
  SEED="${seed}" run_persona_dpo_command \
    recipes/qwen3-1b/dpo/persona/config_dpo_qlora.yaml \
    "${DATASET_DIR}" \
    "qwen3-0.6b-persona12-dpo-s${seed}" \
    "outputs/persona/persona12-dpo-s${seed}" \
    "$@"
done
