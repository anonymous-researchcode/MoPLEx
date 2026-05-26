extra_args=()
if [[ "${ALLOW_TIES:-false}" == "true" ]]; then
  extra_args+=(--allow_ties)
fi
if [[ -n "${MAX_TIES_PER_DIMENSION:-}" ]]; then
  extra_args+=(--max_ties_per_dimension "${MAX_TIES_PER_DIMENSION}")
fi
if [[ -n "${MIN_DISTINCT_RANKINGS:-}" ]]; then
  extra_args+=(--min_distinct_rankings "${MIN_DISTINCT_RANKINGS}")
fi

OUTPUT_DIR="${OUTPUT_DIR:-data/ultrafeedback_disagreement}"

PYTHONPATH=src python scripts/create_disagreement_ultrafeedback_dataset.py \
  --output_dir "${OUTPUT_DIR}" \
  --create_splits \
  "${extra_args[@]}"

# Run this following command
# 
# ALLOW_TIES=true \
# MAX_TIES_PER_DIMENSION=1 \
# MIN_DISTINCT_RANKINGS=4 \
# OUTPUT_DIR=data/ultrafeedback_disagreement \
# bash scripts/bash_scripts/generate_disagreement_ultrafeedback_dataset.sh
# 