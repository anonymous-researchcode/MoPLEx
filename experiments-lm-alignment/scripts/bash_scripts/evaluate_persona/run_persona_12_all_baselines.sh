#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINES="${BASELINES:-dpo listpo}"
export SEEDS="${SEEDS:-42 43}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

for baseline in ${BASELINES}; do
  case "${baseline}" in
    dpo)
      echo "Launching PERSONA-12 DPO: seeds=${SEEDS}"
      bash "${SCRIPT_DIR}/run_persona_12_dpo.sh" "$@"
      ;;
    listpo|listdpo)
      echo "Launching PERSONA-12 ListPO: seeds=${SEEDS}"
      bash "${SCRIPT_DIR}/run_persona_12_listpo.sh" "$@"
      ;;
    *)
      echo "Unknown PERSONA-12 baseline '${baseline}'" >&2
      echo "Valid baselines: dpo listpo" >&2
      exit 1
      ;;
  esac
done
