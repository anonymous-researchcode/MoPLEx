#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/persona_common.sh"

DATASET_DIR="$(persona_pairwise_dataset_dir)"
require_persona_dataset "${DATASET_DIR}" "${TRAIN_NUM_RESPONSES_PER_PROMPT:-${NUM_RESPONSES_PER_PROMPT:-2}}"

run_persona_dpo_command \
  recipes/qwen3-1b/dpo/persona/config_dpo_qlora.yaml \
  "${DATASET_DIR}" \
  "qwen3-0.6b-persona-single-dpo-${RUN_TAG}" \
  "outputs/persona/single-dpo-${RUN_TAG}"
