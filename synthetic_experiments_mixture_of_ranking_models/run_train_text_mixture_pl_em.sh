#!/usr/bin/env bash
set -euo pipefail

# Example script: train text-backbone mixture-of-PL model using EM.

# Optional: activate your environment first, for example:
# conda activate alignment
train_size=3000
lr=5e-5
m_step_updates=20

for m in 2 3 4 5 10
do
args=(
  # Path to generated training dataset (jsonl).
  --train_path mixture_of_pls/synth_data_n12k_m${m}_k3/train.jsonl
  # Path to generated validation dataset (jsonl).
  --val_path mixture_of_pls/synth_data_n12k_m${m}_k3/val.jsonl
  # Directory to save best checkpoint and metric history.
  --output_dir mixture_of_pls/runs/em_n12k_m${m}_k3_qwen3_06b

  # Number of passes over the training dataset.
  --epochs 10
  # Number of examples per optimization batch.
  --batch_size 32
  # If >0 and smaller than full train size, randomly subsample this many train examples.
  # Set 0 to use the full training set.
  --train_subset_size $train_size
  # Learning rate for AdamW.
  --lr $lr
  # Weight decay for AdamW regularization.
  --weight_decay 1e-4
  # Maximum token length used by tokenizer/backbone.
  --max_length 96
  # Random seed for reproducibility.
  --seed 42
  # Compute device: cuda or cpu.
  --device cuda

  # Temperature used in E-step responsibilities gamma.
  --em_temperature 1.0
  # Number of gradient updates performed in each M-step.
  --m_step_updates $m_step_updates
  # If >0, print running train metrics every N batches within an epoch.
  --log_every_batches 10
  # If >0, run validation every N batches (costly on large val sets).
  # Set 0 to disable mid-epoch validation.
  --eval_every_batches 10

  # Enable contextual routing alpha(x).
  # Remove this flag for classic global mixture weights.
  # --use_router

  # Tiny backbone width (embedding/hidden dimension).
  --d_model 128
  # Tiny backbone attention heads.
  --n_heads 4
  # Tiny backbone transformer layers.
  --n_layers 2
  # Tiny backbone dropout probability.
  --dropout 0.1
  # Maximum tokenizer vocabulary size for tiny tokenizer.
  --max_vocab_size 5000

    # # Optional Hugging Face backbone; uncomment to use pretrained text encoder.
    # --hf_model_name "Qwen/Qwen3-0.6B"
    # # Freeze pretrained backbone parameters (only train heads/router/mixture).
    # --freeze_hf_backbone

    # Enable Weights & Biases logging.
    --use_wandb
    # WandB organization/user.
    --wandb_entity "anonymous"
    # WandB project name.
    --wandb_project "multimodal-preference-optimization"
    # Optional run name for easier tracking in the dashboard.
    --wandb_run_name "small-transformers-n12k-m${m}-k3-train${train_size}-lr${lr}-mstep${m_step_updates}"
    # WandB mode: online, offline, or disabled.
    --wandb_mode online

#   # Optional Hugging Face backbone; uncomment to use pretrained text encoder.
#   --hf_model_name "Qwen/Qwen3-0.6B"
#   # Freeze pretrained backbone parameters (only train heads/router/mixture).
#   --freeze_hf_backbone
)

# python
python train_text_mixture_pl_lm.py "${args[@]}"
done
# Notes:
# - Classic EM mode (default): global mixture weights alpha.
# - Contextual MoE mode: add --use_router.
