from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from .listwise_dpo import ListwiseDPODataCollator
from .mixture_pl_components import em_responsibilities, mixture_pl_nll


@dataclass
class ClusterAlignment:
    true_to_pred: dict[int, int]
    pred_to_true: dict[int, int]


class RankingMetricAccumulator:
    def __init__(self) -> None:
        self.total_rows = 0
        self.top1_correct = 0
        self.pairwise_correct = 0
        self.pairwise_total = 0
        self.by_dimension = defaultdict(lambda: {"rows": 0, "top1": 0, "pairwise_correct": 0, "pairwise_total": 0})

    def add_batch(
        self,
        utilities: torch.Tensor,
        candidate_mask: torch.Tensor,
        dimensions: list[str],
        ranked_prefix_lengths: torch.Tensor | None = None,
    ) -> None:
        utilities = utilities.detach().float().cpu()
        candidate_mask = candidate_mask.detach().bool().cpu()
        if ranked_prefix_lengths is None:
            ranked_prefix_lengths = candidate_mask.long().sum(dim=1)
        ranked_prefix_lengths = ranked_prefix_lengths.detach().long().cpu()

        for row_idx, dimension in enumerate(dimensions):
            valid_indices = torch.where(candidate_mask[row_idx])[0].tolist()
            if len(valid_indices) < 2:
                continue

            row_utilities = utilities[row_idx]
            pred_top = max(valid_indices, key=lambda idx: float(row_utilities[idx]))
            top1_correct = int(pred_top == 0)

            pairwise_correct = 0
            pairwise_total = 0
            ranked_prefix_length = int(ranked_prefix_lengths[row_idx].item())
            for left_pos, left_idx in enumerate(valid_indices):
                if left_pos >= ranked_prefix_length:
                    break
                for right_idx in valid_indices[left_pos + 1 :]:
                    pairwise_total += 1
                    if row_utilities[left_idx] > row_utilities[right_idx]:
                        pairwise_correct += 1

            self.total_rows += 1
            self.top1_correct += top1_correct
            self.pairwise_correct += pairwise_correct
            self.pairwise_total += pairwise_total

            dim_stats = self.by_dimension[str(dimension)]
            dim_stats["rows"] += 1
            dim_stats["top1"] += top1_correct
            dim_stats["pairwise_correct"] += pairwise_correct
            dim_stats["pairwise_total"] += pairwise_total

    def metrics(self, prefix: str) -> dict[str, float]:
        metrics = {
            f"{prefix}/num_examples": float(self.total_rows),
            f"{prefix}/top1_acc": self.top1_correct / max(self.total_rows, 1),
            f"{prefix}/pairwise_acc": self.pairwise_correct / max(self.pairwise_total, 1),
        }

        dim_top1_values = []
        dim_pairwise_values = []
        for dimension, stats in sorted(self.by_dimension.items()):
            dim_top1 = stats["top1"] / max(stats["rows"], 1)
            dim_pairwise = stats["pairwise_correct"] / max(stats["pairwise_total"], 1)
            dim_top1_values.append(dim_top1)
            dim_pairwise_values.append(dim_pairwise)
            dim_key = _metric_dimension_key(dimension)
            metrics[f"{prefix}/by_dimension/{dim_key}/num_examples"] = float(stats["rows"])
            metrics[f"{prefix}/by_dimension/{dim_key}/top1_acc"] = dim_top1
            metrics[f"{prefix}/by_dimension/{dim_key}/pairwise_acc"] = dim_pairwise

        metrics[f"{prefix}/macro_top1_acc"] = float(np.mean(dim_top1_values)) if dim_top1_values else 0.0
        metrics[f"{prefix}/macro_pairwise_acc"] = (
            float(np.mean(dim_pairwise_values)) if dim_pairwise_values else 0.0
        )
        return metrics


def _metric_dimension_key(dimension: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(dimension)).strip("_") or "unknown"


def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logps = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_logps = token_logps * valid
    return token_logps.sum(dim=-1)


def _model_ref_logps(trainer, flat_input_ids, flat_attention_mask, flat_labels) -> torch.Tensor:
    if getattr(trainer, "ref_model", None) is not None:
        ref_logits = trainer.ref_model(input_ids=flat_input_ids, attention_mask=flat_attention_mask).logits
        return _sequence_logps(ref_logits, flat_labels)

    model = trainer.model
    adapter_context = nullcontext()
    if hasattr(model, "disable_adapter"):
        adapter_context = model.disable_adapter()
    elif hasattr(trainer, "accelerator"):
        unwrapped = trainer.accelerator.unwrap_model(model)
        if hasattr(unwrapped, "disable_adapter"):
            adapter_context = unwrapped.disable_adapter()

    with adapter_context:
        ref_logits = model(input_ids=flat_input_ids, attention_mask=flat_attention_mask).logits
    return _sequence_logps(ref_logits, flat_labels)


def _policy_utilities(trainer, batch: dict[str, Any], beta: float) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    batch_size, num_candidates, seq_len = input_ids.shape

    flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
    flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)
    flat_labels = labels.view(batch_size * num_candidates, seq_len)

    policy_logits = trainer.model(input_ids=flat_input_ids, attention_mask=flat_attention_mask).logits
    policy_logps = _sequence_logps(policy_logits, flat_labels).view(batch_size, num_candidates)
    ref_logps = _model_ref_logps(trainer, flat_input_ids, flat_attention_mask, flat_labels).view(
        batch_size,
        num_candidates,
    )
    return beta * (policy_logps - ref_logps)


def _dimension_mapping(preference_dimensions: Optional[list[str]], dataset) -> dict[str, int]:
    if preference_dimensions:
        return {dimension: idx for idx, dimension in enumerate(preference_dimensions)}

    dimensions = set()
    for split in dataset.values():
        if "preference_dimension" in split.column_names:
            dimensions.update(str(dimension) for dimension in split["preference_dimension"])
    return {dimension: idx for idx, dimension in enumerate(sorted(dimensions))}


def _true_cluster_tensor(dimensions: list[str], dimension_to_id: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([dimension_to_id[str(dimension)] for dimension in dimensions], dtype=torch.long, device=device)


def _cluster_alignment(true_labels: list[int], pred_labels: list[int], num_clusters: int) -> ClusterAlignment:
    if not true_labels or not pred_labels:
        return ClusterAlignment(true_to_pred={}, pred_to_true={})
    confusion = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label < num_clusters and pred_label < num_clusters:
            confusion[true_label, pred_label] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    true_to_pred = {int(true_idx): int(pred_idx) for true_idx, pred_idx in zip(row_ind, col_ind)}
    pred_to_true = {int(pred_idx): int(true_idx) for true_idx, pred_idx in zip(row_ind, col_ind)}
    return ClusterAlignment(true_to_pred=true_to_pred, pred_to_true=pred_to_true)


def _cluster_metrics_by_dimension(
    true_labels: list[int],
    pred_labels: list[int],
    dimensions: list[str],
    alignment: ClusterAlignment,
    *,
    prefix: str = "mixture/by_dimension",
) -> dict[str, float]:
    if not true_labels or not pred_labels or not dimensions:
        return {}

    metrics: dict[str, float] = {}
    dim_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, dimension in enumerate(dimensions[: len(true_labels)]):
        dim_to_indices[str(dimension)].append(idx)

    for dimension, indices in sorted(dim_to_indices.items()):
        raw_correct = 0
        aligned_correct = 0
        for idx in indices:
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]
            aligned_label = alignment.pred_to_true.get(pred_label, pred_label)
            raw_correct += int(pred_label == true_label)
            aligned_correct += int(aligned_label == true_label)

        dim_key = _metric_dimension_key(dimension)
        denom = max(len(indices), 1)
        metrics[f"{prefix}/{dim_key}/num_examples"] = float(len(indices))
        metrics[f"{prefix}/{dim_key}/cluster_acc"] = aligned_correct / denom
        metrics[f"{prefix}/{dim_key}/cluster_acc_raw"] = raw_correct / denom

    return metrics


def _comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def _ari(true_labels: list[int], pred_labels: list[int]) -> float:
    if len(true_labels) < 2:
        return 0.0
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    pair_counts = Counter(zip(true_labels, pred_labels))
    sum_comb = sum(_comb2(count) for count in pair_counts.values())
    true_comb = sum(_comb2(count) for count in true_counts.values())
    pred_comb = sum(_comb2(count) for count in pred_counts.values())
    total_comb = _comb2(len(true_labels))
    expected = true_comb * pred_comb / max(total_comb, 1.0)
    max_index = 0.5 * (true_comb + pred_comb)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        return 0.0
    return (sum_comb - expected) / denom


def _nmi(true_labels: list[int], pred_labels: list[int]) -> float:
    if not true_labels:
        return 0.0
    total = len(true_labels)
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    pair_counts = Counter(zip(true_labels, pred_labels))

    mi = 0.0
    for (true_label, pred_label), count in pair_counts.items():
        joint = count / total
        true_prob = true_counts[true_label] / total
        pred_prob = pred_counts[pred_label] / total
        mi += joint * math.log(joint / max(true_prob * pred_prob, 1e-12))

    h_true = -sum((count / total) * math.log(count / total) for count in true_counts.values())
    h_pred = -sum((count / total) * math.log(count / total) for count in pred_counts.values())
    denom = math.sqrt(h_true * h_pred)
    return mi / denom if denom > 1e-12 else 0.0


def compute_validation_cluster_alignment(
    trainer,
    ranking_dataset,
    tokenizer,
    training_args,
    dimension_to_id: dict[str, int],
) -> ClusterAlignment | None:
    if "validation" not in ranking_dataset or not hasattr(trainer, "get_batch_mixture_output"):
        return None
    _, cluster_state = evaluate_ranking_split(
        trainer=trainer,
        split_dataset=ranking_dataset["validation"],
        tokenizer=tokenizer,
        training_args=training_args,
        dimension_to_id=dimension_to_id,
        cluster_alignment=None,
        collect_only_cluster_state=True,
    )
    return _cluster_alignment(
        cluster_state["true_labels"],
        cluster_state["posterior_pred_labels"],
        num_clusters=len(dimension_to_id),
    )


def evaluate_ranking_split(
    trainer,
    split_dataset,
    tokenizer,
    training_args,
    dimension_to_id: dict[str, int],
    cluster_alignment: ClusterAlignment | None = None,
    collect_only_cluster_state: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    collator = ListwiseDPODataCollator(
        tokenizer=tokenizer,
        max_length=getattr(training_args, "max_length", 1024),
        max_prompt_length=getattr(training_args, "max_prompt_length", 512),
    )
    dataloader = DataLoader(
        split_dataset,
        batch_size=getattr(training_args, "per_device_eval_batch_size", 8),
        shuffle=False,
        collate_fn=collator,
    )

    device = trainer.args.device
    beta = getattr(trainer, "listwise_beta_override", None) or getattr(trainer.args, "beta", 0.1)
    policy_acc = RankingMetricAccumulator()
    posterior_acc = RankingMetricAccumulator()
    aligned_acc = RankingMetricAccumulator()

    true_labels: list[int] = []
    posterior_pred_labels: list[int] = []
    dimension_labels: list[str] = []
    gamma_entropy_sum = 0.0
    gamma_count = 0
    mixture_nll_sum = 0.0
    mixture_nll_count = 0

    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = trainer._prepare_inputs(batch)
            dimensions = [str(dimension) for dimension in inputs["preference_dimension"]]
            candidate_mask = inputs["candidate_mask"]
            ranked_prefix_lengths = inputs.get("ranked_prefix_length")

            utilities = _policy_utilities(trainer, inputs, beta=beta)
            if not collect_only_cluster_state:
                policy_acc.add_batch(utilities, candidate_mask, dimensions, ranked_prefix_lengths)

            if not hasattr(trainer, "get_batch_mixture_output"):
                continue

            mixture_output = trainer.get_batch_mixture_output(trainer.model, inputs)
            rankings = trainer._observed_rankings(candidate_mask).long()
            nll, comp_logp = mixture_pl_nll(
                mixture_output["router_logits"],
                mixture_output["rewards"],
                rankings,
                candidate_mask=candidate_mask,
                ranked_prefix_lengths=ranked_prefix_lengths,
            )
            gamma = em_responsibilities(
                mixture_output["router_logits"],
                comp_logp,
                temperature=trainer.mixture_config.em_temperature,
            )
            posterior_pred = gamma.argmax(dim=1)
            true_cluster = _true_cluster_tensor(dimensions, dimension_to_id, device=device)

            true_labels.extend(int(label) for label in true_cluster.detach().cpu().tolist())
            posterior_pred_labels.extend(int(label) for label in posterior_pred.detach().cpu().tolist())
            dimension_labels.extend(dimensions)
            gamma_entropy = -(gamma * torch.log(gamma.clamp_min(1e-12))).sum(dim=1)
            gamma_entropy_sum += float(gamma_entropy.sum().detach().cpu().item())
            gamma_count += int(gamma_entropy.numel())
            mixture_nll_sum += float(nll.detach().cpu().item()) * int(candidate_mask.shape[0])
            mixture_nll_count += int(candidate_mask.shape[0])

            if collect_only_cluster_state:
                continue

            row_indices = torch.arange(gamma.shape[0], device=device)
            posterior_rewards = mixture_output["rewards"][row_indices, posterior_pred, :]
            posterior_acc.add_batch(posterior_rewards, candidate_mask, dimensions, ranked_prefix_lengths)

            if cluster_alignment is not None:
                aligned_components = [
                    cluster_alignment.true_to_pred.get(int(true_id), int(pred_id))
                    for true_id, pred_id in zip(true_cluster.detach().cpu().tolist(), posterior_pred.detach().cpu().tolist())
                ]
                aligned_components_tensor = torch.tensor(aligned_components, dtype=torch.long, device=device)
                aligned_rewards = mixture_output["rewards"][row_indices, aligned_components_tensor, :]
                aligned_acc.add_batch(aligned_rewards, candidate_mask, dimensions, ranked_prefix_lengths)

    cluster_state = {
        "true_labels": true_labels,
        "posterior_pred_labels": posterior_pred_labels,
        "dimension_labels": dimension_labels,
    }
    if collect_only_cluster_state:
        return {}, cluster_state

    metrics = policy_acc.metrics("ranking")

    if posterior_pred_labels:
        num_clusters = len(dimension_to_id)
        alignment = cluster_alignment or _cluster_alignment(true_labels, posterior_pred_labels, num_clusters)
        aligned_pred = [alignment.pred_to_true.get(pred_label, pred_label) for pred_label in posterior_pred_labels]
        cluster_correct = sum(int(pred == true) for pred, true in zip(aligned_pred, true_labels))
        raw_correct = sum(int(pred == true) for pred, true in zip(posterior_pred_labels, true_labels))
        pred_counts = Counter(posterior_pred_labels)

        metrics.update(posterior_acc.metrics("mixture_posterior"))
        if cluster_alignment is not None:
            metrics.update(aligned_acc.metrics("mixture_aligned"))
        metrics.update(
            {
                "mixture/cluster_acc": cluster_correct / max(len(true_labels), 1),
                "mixture/cluster_acc_raw": raw_correct / max(len(true_labels), 1),
                "mixture/nmi": _nmi(true_labels, posterior_pred_labels),
                "mixture/ari": _ari(true_labels, posterior_pred_labels),
                "mixture/gamma_entropy": gamma_entropy_sum / max(gamma_count, 1),
                "mixture/nll": mixture_nll_sum / max(mixture_nll_count, 1),
            }
        )
        for cluster_id in range(num_clusters):
            metrics[f"mixture/cluster_balance/{cluster_id}"] = pred_counts.get(cluster_id, 0) / max(
                len(posterior_pred_labels),
                1,
            )
        metrics.update(_cluster_metrics_by_dimension(true_labels, posterior_pred_labels, dimension_labels, alignment))

    return metrics, cluster_state


def evaluate_ranking_splits(trainer, ranking_dataset, tokenizer, script_args, training_args) -> dict[str, dict[str, float]]:
    dimension_to_id = _dimension_mapping(script_args.preference_dimensions, ranking_dataset)
    cluster_alignment = compute_validation_cluster_alignment(
        trainer=trainer,
        ranking_dataset=ranking_dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        dimension_to_id=dimension_to_id,
    )

    results = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in ranking_dataset:
            continue
        split_metrics, _ = evaluate_ranking_split(
            trainer=trainer,
            split_dataset=ranking_dataset[split_name],
            tokenizer=tokenizer,
            training_args=training_args,
            dimension_to_id=dimension_to_id,
            cluster_alignment=cluster_alignment,
        )
        results[split_name] = split_metrics
    return results
