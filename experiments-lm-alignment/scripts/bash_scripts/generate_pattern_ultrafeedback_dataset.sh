PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_pattern_ultrafeedback_dataset.py \
  --dimension_pair instruction_following helpfulness \
  --rank_patterns ABCD BADC \
  --output_dir data/ultrafeedback_pattern_if_help_abcd_badc \
  --create_splits

PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_pattern_ultrafeedback_dataset.py \
  --dimension_pair instruction_following truthfulness \
  --rank_patterns ABCD BADC \
  --output_dir data/ultrafeedback_pattern_if_truth_abcd_badc \
  --create_splits

PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_pattern_ultrafeedback_dataset.py \
  --dimension_pair instruction_following honesty \
  --rank_patterns ABCD BADC \
  --output_dir data/ultrafeedback_pattern_if_honesty_abcd_badc \
  --create_splits

PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_pattern_ultrafeedback_dataset.py \
  --dimension_pair helpfulness honesty \
  --rank_patterns ABCD BADC \
  --output_dir data/ultrafeedback_pattern_help_honesty_abcd_badc \
  --create_splits