#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/persona_common.sh"

MIXTURE_PL_KS="${MIXTURE_PL_KS:-2}"
NUM_CLUSTERS="${NUM_CLUSTERS:-${NUM_PERSONAS}}"

for mixture_pl_k in ${MIXTURE_PL_KS}; do
  DATASET_DIR="$(persona_listwise_dataset_dir "${mixture_pl_k}")"
  require_persona_dataset "${DATASET_DIR}" "${mixture_pl_k}"

  run_persona_dpo_command \
    recipes/qwen3-1b/dpo/persona/config_mixture_pl_top1_qlora.yaml \
    "${DATASET_DIR}" \
    "qwen3-0.6b-persona-mixture-pl-top1-${RUN_TAG}-k${mixture_pl_k}-c${NUM_CLUSTERS}" \
    "outputs/persona/mixture-pl-top1-${RUN_TAG}-k${mixture_pl_k}-c${NUM_CLUSTERS}" \
    --listwise_num_responses "${mixture_pl_k}" \
    --num_clusters "${NUM_CLUSTERS}"
done
