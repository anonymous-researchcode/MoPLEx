"""
Reusable Mixture of Plackett-Luce (MoPL) components for DPO training.

This module provides utilities for mixture-model clustering of rankings, including
mask-aware Plackett-Luce likelihoods and detached-responsibility EM-style losses.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def pl_log_prob(
    rewards: torch.Tensor,
    rankings: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    ranked_prefix_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log probability under Plackett-Luce model.

    Args:
        rewards: [B, M] tensor of item scores
        rankings: [B, M] tensor of ranking indices
        candidate_mask: optional [B, M] mask for valid items
        ranked_prefix_lengths: optional [B] tensor. If provided, only the
            first N ranked positions contribute to the PL likelihood for each
            row. This supports top-1 partial rankings where negatives are
            unordered.

    Returns:
        [B] tensor of log probabilities
    """
    m_items = rewards.shape[1]
    ranked = rewards.gather(1, rankings)
    if candidate_mask is None:
        ranked_mask = torch.ones_like(ranked, dtype=torch.bool)
    else:
        ranked_mask = candidate_mask.gather(1, rankings).bool()

    if ranked_prefix_lengths is None:
        ranked_prefix_lengths = ranked_mask.long().sum(dim=1)
    else:
        ranked_prefix_lengths = ranked_prefix_lengths.to(device=rewards.device, dtype=torch.long)
        ranked_prefix_lengths = ranked_prefix_lengths.clamp(min=0, max=m_items)

    out = torch.zeros(rewards.shape[0], device=rewards.device)
    for t in range(m_items):
        suffix_mask = ranked_mask[:, t:]
        logits = ranked[:, t:].masked_fill(~suffix_mask, float("-inf"))
        term = ranked[:, t] - torch.logsumexp(logits, dim=1)
        contributes = ranked_mask[:, t] & (ranked_prefix_lengths > t)
        out = out + torch.where(contributes, term, torch.zeros_like(term))
    return out


def mixture_pl_nll(
    router_logits: torch.Tensor,
    rewards: torch.Tensor,
    rankings: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    ranked_prefix_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mixture of Plackett-Luce negative log-likelihood and component log-probs.

    Args:
        router_logits: [B, K] logits for mixture components
        rewards: [B, K, M] scores per component per item
        rankings: [B, M] ranking indices
        candidate_mask: optional [B, M] mask for valid items
        ranked_prefix_lengths: optional [B] tensor limiting how many observed
            ranked positions contribute per row.

    Returns:
        nll: scalar negative log-likelihood
        comp_logp: [B, K] log-probabilities under each component
    """
    log_alpha = F.log_softmax(router_logits, dim=1)
    bsz, k_clusters, _ = rewards.shape
    comp_logp = []
    for c in range(k_clusters):
        comp_logp.append(
            pl_log_prob(
                rewards[:, c, :],
                rankings,
                candidate_mask=candidate_mask,
                ranked_prefix_lengths=ranked_prefix_lengths,
            )
        )
    comp_logp = torch.stack(comp_logp, dim=1)
    mix_logp = torch.logsumexp(log_alpha + comp_logp, dim=1)
    if candidate_mask is not None:
        valid_rows = candidate_mask.sum(dim=1) >= 2
        if ranked_prefix_lengths is not None:
            valid_rows = valid_rows & (ranked_prefix_lengths.to(candidate_mask.device) >= 1)
        if not torch.any(valid_rows):
            return rewards.new_zeros(()), comp_logp
        mix_logp = mix_logp[valid_rows]
    nll = -mix_logp.mean()
    return nll, comp_logp


def em_responsibilities(
    router_logits: torch.Tensor, comp_logp: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    E-step: compute soft cluster assignments (posterior probabilities).

    Args:
        router_logits: [B, K] logits for mixture components
        comp_logp: [B, K] log-probabilities under each component
        temperature: temperature for softening assignments (>0)

    Returns:
        gamma: [B, K] posterior probabilities (responsibilities)
    """
    log_alpha = F.log_softmax(router_logits, dim=1)
    joint = log_alpha + comp_logp
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    gamma = F.softmax(joint / temperature, dim=1)
    return gamma


def em_expected_complete_nll(
    router_logits: torch.Tensor,
    comp_logp: torch.Tensor,
    gamma: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    M-step: compute expected complete-data log-likelihood.

    Args:
        router_logits: [B, K] logits for mixture components
        comp_logp: [B, K] log-probabilities under each component
        gamma: [B, K] soft assignments from E-step
        valid_mask: optional [B] mask for examples with enough valid items

    Returns:
        loss: scalar expected NLL (to be minimized)
    """
    log_alpha = F.log_softmax(router_logits, dim=1)
    expected_joint = (gamma * (log_alpha + comp_logp)).sum(dim=1)
    if valid_mask is not None:
        if not torch.any(valid_mask):
            return router_logits.new_zeros(())
        expected_joint = expected_joint[valid_mask]
    return -expected_joint.mean()


class MixtureRouterHead(nn.Module):
    """
    Router network for mixture model: predicts mixture component weights.

    Supports two modes:
    1. Contextual (use_contextual=True): per-instance weights via MLP on context
    2. Global (use_contextual=False): learnable global mixture weights
    """

    def __init__(
        self,
        hidden_size: int,
        num_clusters: int,
        use_contextual: bool = False,
        router_hidden_size: int = 256,
    ):
        """
        Args:
            hidden_size: input context dimension (from LM hidden state)
            num_clusters: number of mixture components (clusters)
            use_contextual: if True, use MLP to compute per-instance weights
            router_hidden_size: hidden dimension of MLP (only used if use_contextual=True)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_clusters = num_clusters
        self.use_contextual = use_contextual

        if use_contextual:
            self.router_net = nn.Sequential(
                nn.Linear(hidden_size, router_hidden_size),
                nn.Tanh(),
                nn.Linear(router_hidden_size, num_clusters),
            )
        else:
            self.global_logits = nn.Parameter(torch.zeros(num_clusters))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, hidden_size] context representations

        Returns:
            router_logits: [B, num_clusters] logits for mixture components
        """
        if self.use_contextual:
            return self.router_net(context)
        else:
            bsz = context.shape[0]
            return self.global_logits.unsqueeze(0).expand(bsz, -1)


class MixturePLHead(nn.Module):
    """Router plus per-component reward heads for mixture PL training."""

    def __init__(
        self,
        hidden_size: int,
        num_clusters: int,
        use_contextual_router: bool = False,
        router_hidden_size: int = 256,
    ):
        super().__init__()
        self.router = MixtureRouterHead(
            hidden_size=hidden_size,
            num_clusters=num_clusters,
            use_contextual=use_contextual_router,
            router_hidden_size=router_hidden_size,
        )
        self.reward_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_clusters)])

    def forward(
        self,
        pooled: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled: [B, M, H] candidate representations
            candidate_mask: optional [B, M] mask for valid candidates

        Returns:
            router_logits: [B, K]
            rewards: [B, K, M]
        """
        if candidate_mask is None:
            context = pooled.mean(dim=1)
        else:
            weights = candidate_mask.to(dtype=pooled.dtype).unsqueeze(-1)
            context = (pooled * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        router_logits = self.router(context)
        rewards = torch.stack([head(pooled).squeeze(-1) for head in self.reward_heads], dim=1)
        return router_logits, rewards


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between two vectors."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


@dataclass
class MixtureEvalMetrics:
    """Evaluation metrics for mixture PL model."""

    nll: float  # Negative log-likelihood
    router_acc: float  # Router accuracy (argmax router_logits vs true cluster)
    posterior_acc_raw: float  # Posterior accuracy before alignment
    posterior_acc_aligned: float  # Posterior accuracy after Hungarian alignment
    aligned_mean_corr: float  # Mean correlation after alignment
    assignment: List[int]  # Cluster assignment permutation


def compute_mixture_eval_metrics(
    model_output: Dict[str, torch.Tensor],
    rankings: torch.Tensor,
    true_clusters: torch.Tensor,
    device: torch.device,
    candidate_mask: Optional[torch.Tensor] = None,
    compute_correlation: bool = False,
    rewards_for_corr: torch.Tensor = None,
    features_for_corr: np.ndarray = None,
) -> MixtureEvalMetrics:
    """
    Compute evaluation metrics for mixture model on a batch.

    Args:
        model_output: dict with 'router_logits' and 'rewards' keys
        rankings: [B, M] ranking indices
        true_clusters: [B] ground-truth cluster labels
        device: torch device
        compute_correlation: whether to compute alignment correlation matrix
        rewards_for_corr: [B, K, M] rewards per component (needed if compute_correlation=True)
        features_for_corr: [B, M, K] features for correlation (needed if compute_correlation=True)

    Returns:
        MixtureEvalMetrics with computed metrics
    """
    router_logits = model_output["router_logits"]
    rewards = model_output.get("rewards")

    # Compute mixture NLL and component log-probs
    nll, comp_logp = mixture_pl_nll(router_logits, rewards, rankings, candidate_mask=candidate_mask)

    # Router accuracy: argmax router_logits vs true_clusters
    pred_cluster = router_logits.argmax(dim=1)
    router_acc = float((pred_cluster == true_clusters).float().mean().item())

    # Posterior cluster prediction: argmax_c log p(z=c, pi|x)
    posterior_logits = F.log_softmax(router_logits, dim=1) + comp_logp
    posterior_pred = posterior_logits.argmax(dim=1)
    posterior_acc_raw = float((posterior_pred == true_clusters).float().mean().item())

    # Alignment via Hungarian matching (if requested)
    k_clusters = router_logits.shape[1]
    if compute_correlation and rewards_for_corr is not None and features_for_corr is not None:
        # Compute correlation matrix: head rewards vs features
        all_rewards = [[] for _ in range(k_clusters)]
        all_features = [[] for _ in range(k_clusters)]

        rewards_np = rewards_for_corr.detach().cpu().numpy()  # [B, K, M]
        for head_idx in range(k_clusters):
            all_rewards[head_idx].append(rewards_np[:, head_idx, :].reshape(-1))
        for feat_idx in range(k_clusters):
            all_features[feat_idx].append(features_for_corr[:, :, feat_idx].reshape(-1))

        corr_mat = np.zeros((k_clusters, k_clusters), dtype=np.float64)
        for h in range(k_clusters):
            rvec = np.concatenate(all_rewards[h], axis=0)
            for f in range(k_clusters):
                fvec = np.concatenate(all_features[f], axis=0)
                corr_mat[h, f] = pearson_corr(rvec, fvec)

        # Hungarian matching to maximize correlation
        row_ind, col_ind = linear_sum_assignment(-corr_mat)
        aligned_corr = float(corr_mat[row_ind, col_ind].mean())

        # Remap predictions via alignment
        pred_all = posterior_pred.detach().cpu().numpy()
        true_all = true_clusters.detach().cpu().numpy()
        mapped_pred = np.array([col_ind[p] for p in pred_all], dtype=np.int64)
        posterior_acc_aligned = float((mapped_pred == true_all).mean())
        assignment = col_ind.tolist()
    else:
        # No correlation computation; use identity alignment
        aligned_corr = 0.0
        posterior_acc_aligned = posterior_acc_raw
        assignment = list(range(k_clusters))

    return MixtureEvalMetrics(
        nll=float(nll.item()),
        router_acc=router_acc,
        posterior_acc_raw=posterior_acc_raw,
        posterior_acc_aligned=posterior_acc_aligned,
        aligned_mean_corr=aligned_corr,
        assignment=assignment,
    )
