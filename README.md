# Overview

This repository contains code for learning mixture of Plackett-Luce models using langauge models experiments. It has two main parts:

- `experiments-lm-alignment/`: language-model alignment code built on the Hugging Face Alignment Handbook, with DPO, listwise DPO, Mixture PL, Mixture BT, EM-DPO, MaxMin, and ranking-evaluation scripts.
- `synthetic_experiments_mixture_of_ranking_models/`: small synthetic mixture-of-Plackett-Luce ranking experiments, including data generation and EM training with either a tiny Transformer encoder or an optional Hugging Face backbone.

## Setup

The alignment package expects Python 3.10+ and GPU-oriented ML dependencies.

```bash
cd experiments-lm-alignment
conda env create -f alignment_environment.yml
conda activate alignment
pip install -e .
huggingface-cli login
```

If you use Weights & Biases logging, also log in with:

```bash
wandb login
```

Most scripts should be run from `experiments-lm-alignment/` with:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
```

## Run LM Experiments

The main training entry point is:

```bash
cd experiments-lm-alignment
accelerate launch \
  --config_file recipes/accelerate_configs/single.yaml \
  scripts/dpo.py \
  --config recipes/qwen3-1b/dpo/ultrafeedback_merged/config_mixture_qlora.yaml
```

You can override YAML settings on the command line. For example:

```bash
accelerate launch \
  --config_file recipes/accelerate_configs/single.yaml \
  scripts/dpo.py \
  --config recipes/qwen3-1b/dpo/ultrafeedback_merged/config_mixture_qlora.yaml \
  --dataset_name ./data/cyclic_ultrafeedback_merged \
  --output_dir outputs/my-mixture-run \
  --run_name my-mixture-run \
  --max_steps 1000 \
  --num_clusters 4
```

Useful recipe directories:

- `recipes/qwen3-1b/dpo/`: Qwen DPO, listwise, and mixture configs.
- `recipes/qwen3-1b/dpo/persona/`: PERSONA listwise and mixture configs.
- `scripts/bash_scripts/evaluate_persona/`: launch scripts for PERSONA experiments.
- `scripts/bash_scripts/evaluate_ultrafeedback_disagreement/`: launch scripts for UltraFeedback disagreement experiments.

### Dataset

`scripts/dpo.py` accepts either a Hugging Face dataset name or a local `datasets.load_from_disk` directory through `--dataset_name`.

Pairwise DPO data should use standard prompt/chosen/rejected-style fields. Listwise data can either be preformatted with:

- `prompt`
- `responses`
- `scores`
- `preference_dimension`

or converted from completion-style data using fields such as `instruction`, `completions`, `response`, `scores`, and `annotations`. For listwise conversion, set:

```bash
--dataset_format listwise
--preference_dimensions instruction_following helpfulness honesty truthfulness
--listwise_num_responses 4
```

Mixture training is enabled with:

```bash
--use_mixture true
--mixture_objective pl
--num_clusters 4
--mixture_training_mode em_only
--mixture_reward_backend head
```

Use `--mixture_reward_backend lora` for per-cluster LoRA reward adapters. The LoRA backend disables gradient checkpointing in `scripts/dpo.py` because adapter switching is not compatible with checkpoint recomputation.

## Run Synthetic Mixture-of-PL Experiments

Generate synthetic JSONL ranking data:

```bash
cd synthetic_experiments_mixture_of_ranking_models
bash run_generate_text_mixture_data.sh
```

This writes datasets like:

```text
mixture_of_pls/synth_data_n12k_m4_k3/train.jsonl
mixture_of_pls/synth_data_n12k_m4_k3/val.jsonl
mixture_of_pls/synth_data_n12k_m4_k3/metadata.json
```

Train the synthetic text mixture model:

```bash
bash run_train_text_mixture_pl_em.sh
```

For a smaller CPU smoke run, use:

```bash
python generate_text_mixture_dataset.py \
  --output_dir mixture_of_pls/smoke_m4_k3 \
  --n_train 200 \
  --n_val 50 \
  --m_items 4 \
  --k_clusters 3 \
  --seed 42

python train_text_mixture_pl_lm.py \
  --train_path mixture_of_pls/smoke_m4_k3/train.jsonl \
  --val_path mixture_of_pls/smoke_m4_k3/val.jsonl \
  --output_dir mixture_of_pls/runs/smoke_m4_k3 \
  --epochs 1 \
  --batch_size 16 \
  --device cpu \
  --no-use_wandb
```

Outputs include `best_model.pt` and `history.jsonl` in the selected run directory.

## Tests

Run the focused mixture-component tests with:

```bash
cd experiments-lm-alignment
PYTHONPATH=src python -m unittest tests.test_mixture_pl_components -v
```

Run all tests with:

```bash
cd experiments-lm-alignment
PYTHONPATH=src pytest tests
```

## Important Files

- `experiments-lm-alignment/scripts/dpo.py`: main DPO/listwise/mixture training entry point.
- `experiments-lm-alignment/src/alignment/listwise_dpo.py`: listwise DPO and mixture trainer implementations.
- `experiments-lm-alignment/src/alignment/mixture_pl_components.py`: Mixture PL likelihood, EM responsibilities, router, and metrics.
- `experiments-lm-alignment/src/alignment/data.py`: dataset loading and pairwise/listwise conversion.
- `synthetic_experiments_mixture_of_ranking_models/generate_text_mixture_dataset.py`: synthetic data generator.
- `synthetic_experiments_mixture_of_ranking_models/train_text_mixture_pl_lm.py`: synthetic EM training loop.
