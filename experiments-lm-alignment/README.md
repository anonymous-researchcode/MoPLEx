# Mixture of Plackett-Luce Models in DPO Training - Implementation Summary

## Overview

Mixture of Plackett-Luce (MoPL) models into the DPO training pipeline. The implementation uses observed listwise rankings with a mask-aware mixture PL objective and a detached-responsibility EM-style training term while jointly optimizing the DPO objective.

## What Was Implemented

### 1. Core Mixture PL Components (`alignment/mixture_pl_components.py`)

Extracted and refactored the Mixture PL utilities from the synthetic experiments:

#### Functions:
- **`pl_log_prob(rewards, rankings, candidate_mask=None)`** — Compute Plackett-Luce log probability
  - Input: [B, M] rewards, [B, M] ranking indices, optional [B, M] valid-candidate mask
  - Output: [B] log probabilities
  
- **`mixture_pl_nll(router_logits, rewards, rankings, candidate_mask=None)`** — Mixture model NLL  
  - Input: [B, K] router logits, [B, K, M] component rewards, [B, M] rankings, optional mask
  - Output: scalar NLL, [B, K] component log-probabilities
  
- **`em_responsibilities(router_logits, comp_logp, temperature)`** — E-step soft assignments
  - Input: [B, K] router logits, [B, K] component log-probs, temperature scalar
  - Output: [B, K] posterior probabilities (soft cluster assignments)
  
- **`em_expected_complete_nll(router_logits, comp_logp, gamma)`** — M-step objective  
  - Input: [B, K] router logits, [B, K] comp log-probs, [B, K] responsibilities
  - Output: scalar loss (expected complete-data negative log-likelihood)

#### Classes:
- **`MixtureRouterHead(hidden_size, num_clusters, use_contextual, router_hidden_size)`** — Router network
  - **Contextual mode**: MLP on LM hidden states (per-instance mixture weights)
  - **Global mode**: Learnable parameters (shared mixture weights)

- **`MixturePLHead(hidden_size, num_clusters, use_contextual_router, router_hidden_size)`** — Trainable mixture module
  - Contains the router and per-component reward heads
  - Attached to the trained model so its parameters are optimized and checkpointed
  
- **`MixtureEvalMetrics`** — Metrics dataclass
  - `nll`, `router_acc`, `posterior_acc_raw`, `posterior_acc_aligned`, `aligned_mean_corr`, `assignment`
  
- **`compute_mixture_eval_metrics(...)`** — Batch evaluation with optional correlation alignment

### 2. Configuration (`alignment/configs.py`)

Added `MixturePLConfig` extending `trl.DPOConfig`:

```python
@dataclass
class MixturePLConfig(trl.DPOConfig):
    use_mixture: bool = False  # Enable mixture modeling
    num_clusters: Optional[int] = None  # Auto-detect from preference_dimensions
    em_temperature: float = 1.0  # Soft/hard assignment control
    mixture_nll_weight: float = 0.1  # DPO_loss + weight * mixture_em_nll
    mixture_training_mode: str = "hybrid_dpo_em"  # "hybrid_dpo_em" or "em_only"
    m_step_updates: int = 1  # M-step optimizer updates per E-step in em_only mode
    mixture_reward_backend: str = "head"  # "head" or "lora"
    use_contextual_router: bool = False  # Contextual vs global router
    router_hidden_size: int = 256  # MLP hidden dimension
    use_auxiliary_dimension_loss: bool = False  # Future: supervised alignment
    log_cluster_metrics: bool = True  # Track cluster accuracy
```

### 3. MixtureDPOTrainer (`alignment/listwise_dpo.py`)

Extended `ListwiseDPOTrainer` with mixture modeling:

#### Key Methods:
- **`__init__(..., mixture_config)`** — Initialize the configured reward backend and attach it to the trained model
  - `mixture_reward_backend="head"`: `MixturePLHead` with explicit per-cluster scalar reward heads
  - `mixture_reward_backend="lora"`: creates one LoRA adapter per cluster and a separate mixture router
  
- **`get_batch_mixture_output(model, batch)`** — Compute router logits and per-component rewards
  - Extracts LM hidden states, pools them
  - Head backend: passes valid-candidate pooled states through the attached mixture head
  - LoRA backend: switches across cluster adapters and uses each adapter's DPO utility as that component's reward
  
- **`get_batch_loss_metrics_with_mixture(model, batch, train_eval)`** — Hybrid loss computation
  - Computes DPO loss via listwise PL
  - Uses the dataset-provided response order as the observed ranking
  - Computes mask-aware mixture NLL for metrics
  - Computes detached-responsibility expected complete-data NLL for training
  - Hybrid loss = DPO_loss + mixture_nll_weight × mixture_em_nll
  - Tracks cluster accuracy vs ground-truth preference_dimension
  
- **`compute_loss(model, inputs, ...)`** — Override to include mixture loss  
  
- **`prediction_step(model, inputs, ...)`** — Override to compute eval metrics with mixture

### 4. MixtureEMDPOTrainer (`alignment/listwise_dpo.py`)

Added an EM-only trainer for classic/generalized EM over the mixture objective:

- Selected with `mixture_training_mode="em_only"`
- E-step: computes fixed responsibilities `gamma` with current parameters
- M-step: runs `m_step_updates` optimizer steps on `mixture_em_nll` only
- Does not add the ordinary DPO loss during EM-only M-steps

#### Training Workflow:
1. Forward pass: Get model logits, router logits, component rewards
2. Compute DPO listwise loss from policy/reference log-probabilities
3. Use observed listwise response order and candidate mask for mixture PL likelihood
4. E-step: Compute detached soft cluster assignments from router logits and component log-probs
5. M-step-style update: Backpropagate hybrid loss (DPO + expected complete-data mixture NLL)
6. Eval: Compute cluster accuracy by comparing router predictions vs ground-truth dimensions

### 5. Entry Point Updates (`scripts/dpo.py`)

- Import `MixtureDPOTrainer`, `MixtureEMDPOTrainer`, and `MixturePLConfig`
- Updated trainer selection logic:
  - If `use_mixture=True` and `mixture_training_mode="em_only"` → `MixtureEMDPOTrainer`
  - If `use_mixture=True` → `MixtureDPOTrainer`
  - Elif `listwise=True` → `ListwiseDPOTrainer`  
  - Else → `DPOTrainer`
- Updated `TrlParser` to use `MixturePLConfig` (inherits from `DPOConfig`)

### 6. Module Exports (`alignment/__init__.py`)

Exported new classes:
- `MixturePLConfig`
- `MixtureDPOTrainer`
- `MixtureEMDPOTrainer`

## Usage Example

### Command Line

```bash
python scripts/dpo.py \
    --use_mixture \
    --num_clusters 2 \
    --mixture_nll_weight 0.1 \
    --em_temperature 1.0 \
    --use_contextual_router \
    --router_hidden_size 256 \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_format listwise \
    --preference_dimensions '["helpfulness", "harmlessness"]' \
    --output_dir ./outputs/mixture_dpo_qwen
```

### Programmatic

```python
from alignment import MixturePLConfig, MixtureDPOTrainer, get_dataset

# Configuration
mixture_config = MixturePLConfig(
    use_mixture=True,
    num_clusters=2,
    mixture_nll_weight=0.1,
    em_temperature=1.0,
    use_contextual_router=True,
)

# Trainer
trainer = MixtureDPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    mixture_config=mixture_config,
)

# Training
trainer.train()
```

## Testing

### Unit Tests (19 passing tests)

Location: `experiments-lm-alignment/tests/test_mixture_pl_components.py`

Test Coverage:
- ✅ `TestPlLogProb` — PL log probability shape, gradients, values
- ✅ `TestMixturePLNLL` — Mixture NLL shape and gradients
- ✅ `TestEMResponsibilities` — Responsibility shape, summation, temperature effect
- ✅ `TestEMExpectedCompleteNLL` — M-step loss computation and descent
- ✅ `TestMixtureRouterHead` — Router shapes (contextual & global), gradients, consistency
- ✅ `TestPearsonCorr` — Correlation computation (perfect, negative, zero variance)
- ✅ `TestIntegration` — Full EM loop, training loop simulation

Run tests:
```bash
cd experiments-lm-alignment
PYTHONPATH=src python -m unittest tests.test_mixture_pl_components -v
```

## Key Design Decisions

1. **Mixture Head Ownership**: Router and per-cluster reward heads live in `MixturePLHead`
   - Attached to the trained model so optimizer, Accelerate, and checkpointing see the parameters

2. **Observed Rankings**: The dataset-provided listwise response order is the mixture target
   - Candidate masks are passed into PL likelihoods so padded responses do not affect the loss

3. **Contextual vs Global Router**: Both modes supported
   - Contextual (default): More expressive, per-instance cluster assignment
   - Global: Simpler, shared mixture weights

4. **EM-Style Objective**: Single stochastic E-step per batch
   - Responsibilities are detached before the expected complete-data NLL is optimized

5. **Cluster Metrics**: Use ground-truth preference_dimension for eval-time accuracy
   - Pro: Meaningful evaluation; enables cluster alignment via Hungarian matching
   - Con: Requires dimension labels in data

## Known Limitations & Future Work

1. **Auxiliary Dimension Loss** — Defined but not yet implemented
   - Could add supervised loss term to align router with ground-truth dimensions during training
   - Flag: `use_auxiliary_dimension_loss`

2. **Pairwise DPO Support** — Currently listwise-only
   - Converting pairwise (chosen/rejected) to full rankings is non-trivial
   - Would need preference transitivity modeling

3. **Computational Cost** — Not yet profiled on large datasets
   - Mixture head forward pass and component likelihoods add overhead
   - Recommendations: batch caching of component log-probs if profiling shows this dominates

4. **Correlation Matrix** — Currently optional; not integrated into main loss
   - Could optionally add correlation alignment loss for unsupervised cluster discovery

## Performance Metrics Logged

During training, MixtureDPOTrainer logs:

- **DPO metrics** (from listwise trainer):
  - `eval_listwise/top1_acc` — Top-1 ranking accuracy
  - `eval_listwise/pairwise_acc` — Pairwise ranking accuracy
  - `eval_listwise/utility_*` — Utility statistics

- **Mixture metrics** (new):
  - `mixture/nll` — Mixture PL negative log-likelihood
  - `mixture/cluster_acc` — Cluster assignment accuracy vs ground-truth dimension

All metrics are gathered and averaged across all processes in distributed training.

## Backward Compatibility

✅ **Fully backward compatible**:
- Default `use_mixture=False` preserves original ListwiseDPOTrainer behavior
- Existing pairwise/listwise DPO scripts work unchanged
- No modifications to data loading or preprocessing

## File Changes Summary

| File | Changes |
|------|---------|
| `src/alignment/mixture_pl_components.py` | ✨ NEW — Core mixture PL utilities |
| `src/alignment/configs.py` | ✏️ ADDED `MixturePLConfig` class |
| `src/alignment/listwise_dpo.py` | ✏️ ADDED `MixtureDPOTrainer` class, updated imports |
| `src/alignment/__init__.py` | ✏️ ADDED exports for new classes |
| `scripts/dpo.py` | ✏️ UPDATED trainer selection, imports, parser |
| `tests/test_mixture_pl_components.py` | ✨ NEW — 19 unit tests |

## Verification Checklist

- ✅ All files compile without syntax errors
- ✅ 19 unit tests pass (pl_log_prob, mixture_nll, em_steps, router, integration tests)
- ✅ New classes properly exported from alignment module
- ✅ dpo.py successfully imports and instantiates MixtureDPOTrainer
- ✅ Backward compatibility preserved (use_mixture=False by default)
- ✅ Configuration validates correctly
- ✅ No breaking changes to existing ListwiseDPOTrainer or DPOTrainer

## Next Steps (Phase 5 Continuation)

1. **Integration Test** — Create toy listwise dataset and run full training loop
2. **Real Dataset Test** — Run on UltraFeedback subset with --dataset_format listwise
3. **Metric Validation** — Verify cluster accuracy improves over epochs
4. **Profiling** — Measure computational overhead on realistic batch sizes
5. **Documentation** — Add example configs and training recipes
6. **Optional Enhancements**:
   - Implement auxiliary dimension loss
   - Add pairwise DPO support
   - Optimize component head initialization
   - Add correlation-based unsupervised clustering loss
