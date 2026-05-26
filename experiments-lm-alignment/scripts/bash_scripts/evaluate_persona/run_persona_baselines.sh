#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINES="${BASELINES:-${METHODS:-dpo listdpo mixture_pl}}"

for baseline in ${BASELINES}; do
  case "${baseline}" in
    dpo|single_dpo)
      bash "${SCRIPT_DIR}/run_persona_dpo.sh"
      ;;
    listdpo|single_listdpo)
      bash "${SCRIPT_DIR}/run_persona_listdpo.sh"
      ;;
    mixture_pl)
      bash "${SCRIPT_DIR}/run_persona_mixture_pl.sh"
      ;;
    mixture_bt)
      echo "Baseline 'mixture_bt' is deprecated for PERSONA runs." >&2
      echo "Use BASELINES=mixture_pl MIXTURE_PL_KS=2 for the BT-equivalent mixture PL baseline." >&2
      exit 1
      ;;
    *)
      echo "Unknown PERSONA baseline '${baseline}'" >&2
      echo "Valid baselines: dpo listdpo mixture_pl" >&2
      exit 1
      ;;
  esac
done
