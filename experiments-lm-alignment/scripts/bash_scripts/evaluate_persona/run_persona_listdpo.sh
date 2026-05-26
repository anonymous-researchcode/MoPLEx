#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/persona_common.sh"

LISTWISE_K="${LISTWISE_K:-${NUM_RESPONSES_PER_PROMPT:-2}}"
DATASET_DIR="$(persona_listwise_dataset_dir "${LISTWISE_K}")"
require_persona_dataset "${DATASET_DIR}" "${LISTWISE_K}"

run_persona_dpo_command \
  recipes/qwen3-1b/dpo/persona/config_top1_listwise_qlora.yaml \
  "${DATASET_DIR}" \
  "qwen3-0.6b-persona-single-top1-listdpo-${RUN_TAG}-k${LISTWISE_K}" \
  "outputs/persona/single-top1-listdpo-${RUN_TAG}-k${LISTWISE_K}" \
  --listwise_num_responses "${LISTWISE_K}"
