# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gc
import inspect
import math
import os
import re
import time
from typing import Any, Dict, Literal, Optional, Tuple, Union

from accelerate import PartialState
from datasets import Dataset, IterableDataset
from transformers import BaseImageProcessor, FeatureExtractionMixin, PreTrainedTokenizerBase, ProcessorMixin
from transformers.trainer_utils import TrainOutput
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from trl import DPOTrainer

from .mixture_pl_components import (
    em_expected_complete_nll,
    em_responsibilities,
    mixture_pl_nll,
    MixturePLHead,
    MixtureRouterHead,
    pl_log_prob,
)


def _peft_bf16_autocast_context(trainer) -> Any:
    if getattr(trainer, "_peft_has_been_casted_to_bf16", False):
        return torch.autocast(trainer.accelerator.device.type)
    return nullcontext()


@dataclass
class ListwiseDPODataCollator:
    tokenizer: Any
    max_length: int
    max_prompt_length: int

    def _truncate_prompt(self, prompt_ids: list[int]) -> list[int]:
        if len(prompt_ids) <= self.max_prompt_length:
            return prompt_ids
        return prompt_ids[-self.max_prompt_length :]

    def _build_sequence(self, prompt: str, response: str) -> tuple[list[int], list[int], list[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = self._truncate_prompt(prompt_ids)
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
            response_ids = response_ids + [eos_id]

        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            original_prompt_len = len(prompt_ids)
            if overflow < original_prompt_len:
                prompt_ids = prompt_ids[overflow:]
            else:
                prompt_ids = []
                response_ids = response_ids[overflow - original_prompt_len :]
            input_ids = prompt_ids + response_ids

        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + response_ids
        return input_ids, attention_mask, labels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_candidate_mask = []
        batch_preference_dimensions = []
        batch_ranked_prefix_lengths = []

        max_candidates = max(len(feature["responses"]) for feature in features)

        for feature in features:
            prompt = feature["prompt"]
            responses = feature["responses"]
            batch_preference_dimensions.append(feature.get("preference_dimension", "unknown"))
            ranked_prefix_length = feature.get("ranked_prefix_length", len(responses))
            ranked_prefix_length = max(0, min(int(ranked_prefix_length), len(responses)))
            batch_ranked_prefix_lengths.append(ranked_prefix_length)

            item_input_ids = []
            item_attention_mask = []
            item_labels = []
            item_candidate_mask = [1] * len(responses) + [0] * (max_candidates - len(responses))

            for response in responses:
                input_ids, attention_mask, labels = self._build_sequence(prompt, response)
                item_input_ids.append(input_ids)
                item_attention_mask.append(attention_mask)
                item_labels.append(labels)

            for _ in range(max_candidates - len(responses)):
                item_input_ids.append([])
                item_attention_mask.append([])
                item_labels.append([])

            batch_input_ids.append(item_input_ids)
            batch_attention_mask.append(item_attention_mask)
            batch_labels.append(item_labels)
            batch_candidate_mask.append(item_candidate_mask)

        max_seq_len = max(len(seq) for item in batch_input_ids for seq in item)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for listwise collation.")

        for i in range(len(batch_input_ids)):
            for j in range(len(batch_input_ids[i])):
                seq_len = len(batch_input_ids[i][j])
                pad_len = max_seq_len - seq_len
                batch_input_ids[i][j] = batch_input_ids[i][j] + [pad_id] * pad_len
                batch_attention_mask[i][j] = batch_attention_mask[i][j] + [0] * pad_len
                batch_labels[i][j] = batch_labels[i][j] + [-100] * pad_len

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "candidate_mask": torch.tensor(batch_candidate_mask, dtype=torch.bool),
            "preference_dimension": batch_preference_dimensions,
            "ranked_prefix_length": torch.tensor(batch_ranked_prefix_lengths, dtype=torch.long),
        }


class ListwiseDPOTrainer(DPOTrainer):
    def __init__(self, *args, listwise_beta: float | None = None, **kwargs):
        self.listwise_beta_override = listwise_beta
        if kwargs.get("processing_class") is None and kwargs.get("tokenizer") is not None:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        elif kwargs.get("tokenizer") is not None:
            kwargs.pop("tokenizer")
        if kwargs.get("data_collator") is None:
            processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
            if processing_class is None:
                raise ValueError("A tokenizer/processing_class is required for listwise data collation.")
            max_length = kwargs.get("max_length")
            if max_length is None and kwargs.get("args") is not None:
                max_length = getattr(kwargs["args"], "max_length", None)
            if max_length is None:
                max_length = 1024
            max_prompt_length = kwargs.get("max_prompt_length")
            if max_prompt_length is None and kwargs.get("args") is not None:
                max_prompt_length = getattr(kwargs["args"], "max_prompt_length", None)
            if max_prompt_length is None:
                max_prompt_length = 512
            kwargs["data_collator"] = ListwiseDPODataCollator(
                tokenizer=processing_class,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
            )
        super().__init__(*args, **kwargs)
        self._stored_metrics = {"train": {}, "eval": {}}

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        if not hasattr(self, "_stored_metrics"):
            self._stored_metrics = {"train": {}, "eval": {}}
        bucket = self._stored_metrics.setdefault(train_eval, {})
        for key, value in metrics.items():
            bucket.setdefault(key, []).append(value)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        if not hasattr(self, "_stored_metrics"):
            self._stored_metrics = {"train": {}, "eval": {}}
        self._stored_metrics.setdefault("train", {})
        self._stored_metrics.setdefault("eval", {})
        stored = self._stored_metrics.get(train_eval, {})
        prefix = "eval_" if train_eval == "eval" else ""
        for key, values in stored.items():
            if not values:
                continue
            metric_key = key if key.startswith(prefix) else f"{prefix}{key}"
            logs[metric_key] = torch.tensor(values, dtype=torch.float32).mean().item()
        self._stored_metrics[train_eval] = {}
        try:
            result = super().log(logs, start_time=start_time)
        except TypeError:
            result = super().log(logs)
        self._stored_metrics.setdefault("train", {})
        self._stored_metrics.setdefault("eval", {})
        return result

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "labels",
                "candidate_mask",
                "ranked_prefix_length",
            ]

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: Any,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        """
        For listwise datasets, skip pairwise-specific tokenization and column removal.
        Listwise datasets already have prompt/responses structure; collator handles sequencing.
        """
        from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
        
        with PartialState().main_process_first():
            if isinstance(dataset, Dataset):
                map_kwargs = {"num_proc": args.dataset_num_proc, "writer_batch_size": 10}
            else:
                map_kwargs = {}

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": getattr(args, "tools", None)},
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100

        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_logps = F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * valid
        return token_logps.sum(dim=-1)

    def _forward_ref(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.ref_model is None:
                with self.model.disable_adapter():
                    ref_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
        return self._sequence_logps(ref_logits, labels)

    def _forward_ref_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ref_model = self.model if self.ref_model is None else self.ref_model
        ref_params = tuple(ref_model.parameters())
        original_requires_grad = tuple(param.requires_grad for param in ref_params)
        try:
            for param in ref_params:
                param.requires_grad_(False)
            if self.ref_model is None:
                with self.model.disable_adapter():
                    ref_logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            else:
                ref_logits = self.ref_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        finally:
            for param, requires_grad in zip(ref_params, original_requires_grad):
                param.requires_grad_(requires_grad)
        return self._sequence_logps(ref_logits, labels)

    def _use_linear_reward_approx(self, train_eval: Literal["train", "eval"]) -> bool:
        if not getattr(self.args, "use_linear_reward_approx", False):
            return False
        if train_eval != "train" and getattr(self.args, "linear_approx_exact_eval", True):
            return False
        return True

    def _select_linear_approx_anchors(self, candidate_mask: torch.Tensor) -> list[torch.Tensor]:
        num_anchors = int(getattr(self.args, "linear_approx_num_anchors", 2))
        anchor_positions = []
        for row_mask in candidate_mask:
            valid_positions = torch.nonzero(row_mask, as_tuple=False).flatten()
            if valid_positions.numel() > num_anchors:
                permutation = torch.randperm(valid_positions.numel(), device=valid_positions.device)
                valid_positions = valid_positions.index_select(0, permutation[:num_anchors]).sort().values
            anchor_positions.append(valid_positions)
        return anchor_positions

    @staticmethod
    def _linear_approx_from_anchor_tensors(
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        anchor_positions: list[torch.Tensor],
        anchor_scores: torch.Tensor,
        anchor_grads: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate candidate scores from exact anchor scores and stop-gradient input gradients.

        Shapes:
            candidate_embeddings: [B, M, S, H]
            anchor_scores: [N] or [N, K]
            anchor_grads: [N, S, H] or [N, K, S, H]
        """
        batch_size, num_candidates = candidate_mask.shape
        if anchor_scores.dim() == 1:
            estimates = anchor_scores.new_zeros((batch_size, num_candidates))
        elif anchor_scores.dim() == 2:
            estimates = anchor_scores.new_zeros((batch_size, anchor_scores.shape[1], num_candidates))
        else:
            raise ValueError("anchor_scores must have shape [N] or [N, K]")

        offset = 0
        for row_idx, row_anchor_positions in enumerate(anchor_positions):
            num_row_anchors = row_anchor_positions.numel()
            if num_row_anchors == 0:
                continue

            row_scores = anchor_scores[offset : offset + num_row_anchors]
            row_grads = anchor_grads[offset : offset + num_row_anchors]
            row_embeddings = candidate_embeddings[row_idx].to(dtype=row_grads.dtype)
            anchor_embeddings = row_embeddings.index_select(0, row_anchor_positions)
            deltas = row_embeddings.unsqueeze(0) - anchor_embeddings.unsqueeze(1)

            if anchor_scores.dim() == 1:
                dot_terms = (row_grads.unsqueeze(1) * deltas).sum(dim=(-1, -2))
                row_estimates = (row_scores.detach().unsqueeze(1) + dot_terms).mean(dim=0)
                row_estimates[row_anchor_positions] = row_scores
                estimates[row_idx] = row_estimates.to(dtype=estimates.dtype)
            else:
                dot_terms = (row_grads.unsqueeze(2) * deltas.unsqueeze(1)).sum(dim=(-1, -2))
                row_estimates = (row_scores.detach().unsqueeze(-1) + dot_terms).mean(dim=0)
                row_estimates[:, row_anchor_positions] = row_scores.transpose(0, 1)
                estimates[row_idx] = row_estimates.to(dtype=estimates.dtype)

            offset += num_row_anchors

        if anchor_scores.dim() == 1:
            return estimates.masked_fill(~candidate_mask, 0.0)
        return estimates.masked_fill(~candidate_mask.unsqueeze(1), 0.0)

    def _approximate_listwise_utilities(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)
        flat_labels = labels.view(batch_size * num_candidates, seq_len)

        embedding_layer = model.get_input_embeddings()
        candidate_embeddings = embedding_layer(flat_input_ids).detach().view(batch_size, num_candidates, seq_len, -1)
        anchor_positions = self._select_linear_approx_anchors(candidate_mask)
        anchor_flat_indices = [
            row_idx * num_candidates + position
            for row_idx, positions in enumerate(anchor_positions)
            for position in positions.tolist()
        ]
        if not anchor_flat_indices:
            return input_ids.new_zeros((batch_size, num_candidates), dtype=torch.float32)

        anchor_flat_indices_tensor = torch.tensor(anchor_flat_indices, device=input_ids.device, dtype=torch.long)
        anchor_embeds = candidate_embeddings.view(batch_size * num_candidates, seq_len, -1).index_select(
            0, anchor_flat_indices_tensor
        )
        anchor_embeds = anchor_embeds.detach().requires_grad_(True)
        anchor_attention_mask = flat_attention_mask.index_select(0, anchor_flat_indices_tensor)
        anchor_labels = flat_labels.index_select(0, anchor_flat_indices_tensor)
        ref_mode = getattr(self.args, "linear_approx_ref_mode", "input_gradient")
        beta = self.listwise_beta_override if self.listwise_beta_override is not None else self.beta
        exact_ref_logps = None
        if ref_mode == "exact_score":
            exact_ref_logps = self._forward_ref(flat_input_ids, flat_attention_mask, flat_labels).view(
                batch_size,
                num_candidates,
            )

        with torch.enable_grad():
            policy_logits = model(inputs_embeds=anchor_embeds, attention_mask=anchor_attention_mask).logits
            policy_logps = self._sequence_logps(policy_logits, anchor_labels)
            if ref_mode == "exact_score":
                anchor_grads = torch.autograd.grad(
                    policy_logps.sum(),
                    anchor_embeds,
                    retain_graph=True,
                    create_graph=False,
                )[0].detach()
                anchor_scores = policy_logps
            else:
                ref_logps = self._forward_ref_from_embeds(anchor_embeds, anchor_attention_mask, anchor_labels)
                anchor_scores = beta * (policy_logps - ref_logps)
                anchor_grads = torch.autograd.grad(
                    anchor_scores.sum(),
                    anchor_embeds,
                    retain_graph=True,
                    create_graph=False,
                )[0].detach()

        estimates = self._linear_approx_from_anchor_tensors(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_scores,
            anchor_grads,
        )
        if ref_mode == "exact_score":
            return beta * (estimates - exact_ref_logps.detach())
        return estimates

    @staticmethod
    def _pl_negative_log_likelihood(
        utilities: torch.Tensor,
        candidate_mask: torch.Tensor,
        ranked_prefix_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Mask padded candidates and compute suffix logsumexp in parallel.
        masked_utilities = utilities.masked_fill(~candidate_mask, float("-inf"))
        suffix_lse = torch.flip(torch.logcumsumexp(torch.flip(masked_utilities, dims=[1]), dim=1), dims=[1])

        # Per-position PL term: -(u_k - logsumexp(u_k, ..., u_m)), only on valid candidates.
        if ranked_prefix_lengths is None:
            ranked_prefix_lengths = candidate_mask.long().sum(dim=1)
        else:
            ranked_prefix_lengths = ranked_prefix_lengths.to(device=utilities.device, dtype=torch.long)
            ranked_prefix_lengths = ranked_prefix_lengths.clamp(min=0, max=utilities.shape[1])

        positions = torch.arange(utilities.shape[1], device=utilities.device).unsqueeze(0)
        observed_rank_mask = candidate_mask & (positions < ranked_prefix_lengths.unsqueeze(1))
        diff = torch.where(observed_rank_mask, masked_utilities - suffix_lse, torch.zeros_like(utilities))
        row_losses = (-diff).sum(dim=1)

        valid_rows = (candidate_mask.sum(dim=1) >= 2) & (ranked_prefix_lengths >= 1)
        if not torch.any(valid_rows):
            return utilities.new_zeros(())
        return row_losses[valid_rows].mean()

    @staticmethod
    def _metric_dimension_key(dimension: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dimension).strip()).strip("_")
        return normalized or "unknown"

    @staticmethod
    def _listwise_metrics(
        utilities: torch.Tensor,
        candidate_mask: torch.Tensor,
        ranked_prefix_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        masked_utilities = utilities.masked_fill(~candidate_mask, float("-inf"))
        pred_order = torch.argsort(masked_utilities, dim=1, descending=True)
        top1_acc = (pred_order[:, 0] == 0).float().mean()

        num_candidates = utilities.shape[1]
        if ranked_prefix_lengths is None:
            ranked_prefix_lengths = candidate_mask.long().sum(dim=1)
        else:
            ranked_prefix_lengths = ranked_prefix_lengths.to(device=utilities.device, dtype=torch.long)
            ranked_prefix_lengths = ranked_prefix_lengths.clamp(min=0, max=num_candidates)

        positions = torch.arange(num_candidates, device=utilities.device)
        observed_left = positions.unsqueeze(0) < ranked_prefix_lengths.unsqueeze(1)
        upper_tri = torch.triu(
            torch.ones((num_candidates, num_candidates), dtype=torch.bool, device=utilities.device),
            diagonal=1,
        )
        valid_pairs = (
            candidate_mask.unsqueeze(2)
            & candidate_mask.unsqueeze(1)
            & upper_tri.unsqueeze(0)
            & observed_left.unsqueeze(2)
        )
        pairwise_correct = (masked_utilities.unsqueeze(2) > masked_utilities.unsqueeze(1)) & valid_pairs

        total_pairs = valid_pairs.sum(dim=(1, 2))
        correct_pairs = pairwise_correct.sum(dim=(1, 2))
        row_pairwise_acc = correct_pairs.float() / total_pairs.clamp_min(1).float()
        valid_rows = total_pairs > 0
        pairwise_acc = row_pairwise_acc[valid_rows].mean() if torch.any(valid_rows) else utilities.new_zeros(())

        utility_first = torch.where(
            candidate_mask[:, 0],
            utilities[:, 0],
            torch.zeros_like(utilities[:, 0]),
        ).mean()

        valid_counts = candidate_mask.sum(dim=1)
        valid_utility_rows = valid_counts > 0
        if torch.any(valid_utility_rows):
            last_indices = valid_counts[valid_utility_rows] - 1
            row_indices = torch.arange(last_indices.shape[0], device=utilities.device)
            utility_last = utilities[valid_utility_rows][row_indices, last_indices].mean()
        else:
            utility_last = utilities.new_zeros(())

        return {
            "listwise/top1_acc": top1_acc,
            "listwise/pairwise_acc": pairwise_acc,
            "listwise/utility_first": utility_first,
            "listwise/utility_last": utility_last,
            "listwise/utility_mean": utilities.mean(),
        }

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Any],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        candidate_mask = batch["candidate_mask"]
        ranked_prefix_lengths = batch.get("ranked_prefix_length")
        preference_dimensions = batch.get("preference_dimension")

        batch_size, num_candidates, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)
        flat_labels = labels.view(batch_size * num_candidates, seq_len)

        if self._use_linear_reward_approx(train_eval):
            utilities = self._approximate_listwise_utilities(
                model,
                input_ids,
                attention_mask,
                labels,
                candidate_mask,
            )
        else:
            policy_logits = model(input_ids=flat_input_ids, attention_mask=flat_attention_mask).logits
            policy_logps = self._sequence_logps(policy_logits, flat_labels).view(batch_size, num_candidates)
            ref_logps = self._forward_ref(flat_input_ids, flat_attention_mask, flat_labels).view(batch_size, num_candidates)

            beta = self.listwise_beta_override if self.listwise_beta_override is not None else self.beta
            utilities = beta * (policy_logps - ref_logps)
        loss = self._pl_negative_log_likelihood(utilities, candidate_mask, ranked_prefix_lengths)

        metric_tensors = self._listwise_metrics(utilities, candidate_mask, ranked_prefix_lengths)

        if preference_dimensions is not None:
            dim_to_indices: dict[str, list[int]] = {}
            for idx, dimension in enumerate(preference_dimensions):
                dim_key = self._metric_dimension_key(dimension)
                dim_to_indices.setdefault(dim_key, []).append(idx)

            for dim_key, indices in dim_to_indices.items():
                dim_index_tensor = torch.tensor(indices, device=utilities.device, dtype=torch.long)
                dim_metrics = self._listwise_metrics(
                    utilities.index_select(0, dim_index_tensor),
                    candidate_mask.index_select(0, dim_index_tensor),
                    ranked_prefix_lengths.index_select(0, dim_index_tensor)
                    if ranked_prefix_lengths is not None
                    else None,
                )
                for metric_name, metric_value in dim_metrics.items():
                    suffix = metric_name.removeprefix("listwise/")
                    metric_tensors[f"listwise/by_dimension/{dim_key}/{suffix}"] = metric_value

        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}{name}": self.accelerator.gather_for_metrics(value.detach()).mean().item()
            for name, value in metric_tensors.items()
        }

        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        compute_loss_context_manager = (
            _peft_bf16_autocast_context(self)
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        del ignore_keys
        prediction_context_manager = (
            _peft_bf16_autocast_context(self)
        )
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = torch.tensor(
            [
                metrics["eval_listwise/utility_first"],
                metrics["eval_listwise/utility_last"],
            ],
            device=self.accelerator.device,
        )
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)


class MixtureDPOTrainer(ListwiseDPOTrainer):
    """
    Listwise DPO trainer with Mixture of Plackett-Luce (MoPL) clustering.

    Jointly optimizes DPO loss and mixture PL loss to discover latent ranking clusters
    while learning dimension-aware preferences.
    """

    def __init__(
        self,
        model,
        ref_model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        mixture_config=None,
        **kwargs,
    ):
        """
        Args:
            mixture_config: MixturePLConfig instance with mixture hyperparameters
        """
        if tokenizer is not None and kwargs.get("processing_class") is None:
            kwargs["processing_class"] = tokenizer

        dpo_init_kwargs = dict(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            **kwargs,
        )
        supported_init_args = set(inspect.signature(DPOTrainer.__init__).parameters)
        dpo_init_kwargs = {key: value for key, value in dpo_init_kwargs.items() if key in supported_init_args}
        super().__init__(**dpo_init_kwargs)

        self.mixture_config = mixture_config
        if mixture_config is None or not mixture_config.use_mixture:
            raise ValueError("MixtureDPOTrainer requires mixture_config with use_mixture=True")

        # Determine number of clusters
        self.num_clusters = mixture_config.num_clusters
        if self.num_clusters is None:
            # Try to infer from preference_dimensions in script args
            if hasattr(args, "preference_dimensions") and args.preference_dimensions:
                self.num_clusters = len(args.preference_dimensions)
            else:
                raise ValueError(
                    "num_clusters must be specified in mixture_config or inferred from preference_dimensions"
                )

        # Get hidden size from the trained model after DPOTrainer has applied any wrapping.
        model_config = getattr(self.model, "config", getattr(model, "config", None))
        if model_config is not None and hasattr(model_config, "hidden_size"):
            self.hidden_size = model_config.hidden_size
        else:
            raise ValueError("Model config must have hidden_size attribute")

        self.mixture_reward_backend = mixture_config.mixture_reward_backend
        self.mixture_head = None
        self.mixture_router = None
        self.policy_adapter_name = None
        self.mixture_adapter_names = []

        if self.mixture_reward_backend == "head":
            self.mixture_head = MixturePLHead(
                hidden_size=self.hidden_size,
                num_clusters=self.num_clusters,
                use_contextual_router=mixture_config.use_contextual_router,
                router_hidden_size=mixture_config.router_hidden_size,
            )
            if (
                not mixture_config.use_contextual_router
                and getattr(mixture_config, "use_closed_form_router_prior_update", False)
                and hasattr(self.mixture_head.router, "global_logits")
            ):
                self.mixture_head.router.global_logits.requires_grad_(False)
            self.mixture_head.to(self.args.device)
            self.model.add_module("mixture_pl_head", self.mixture_head)
        else:
            self._init_lora_mixture_backend(peft_config)

        # Kept for quick diagnostics in notebooks/scripts.
        self.router_accs_buffer = []
        self.preference_dimension_to_cluster = self._infer_preference_dimension_to_cluster(
            self.train_dataset,
            self.eval_dataset,
        )

    def _get_active_adapter_name(self) -> Optional[str]:
        adapter_model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        active_adapter = getattr(adapter_model, "active_adapter", None)
        if isinstance(active_adapter, str):
            return active_adapter
        active_adapters = getattr(adapter_model, "active_adapters", None)
        if isinstance(active_adapters, list) and active_adapters:
            return str(active_adapters[0])
        return None

    def _set_active_adapter(self, adapter_name: str) -> None:
        adapter_model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        if not hasattr(adapter_model, "set_adapter"):
            raise ValueError("LoRA mixture backend requires a PEFT model with set_adapter().")
        adapter_model.set_adapter(adapter_name)

    def _peft_adapter_config(self, peft_config):
        if peft_config is not None:
            return peft_config
        adapter_model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        model_peft_config = getattr(adapter_model, "peft_config", None)
        if not model_peft_config:
            return None
        active_adapter = self._get_active_adapter_name()
        if active_adapter is not None and active_adapter in model_peft_config:
            return model_peft_config[active_adapter]
        return next(iter(model_peft_config.values()))

    def _set_trainable_lora_adapters(self, adapter_names: list[str]) -> None:
        adapter_model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        adapter_tokens = tuple(f".{adapter_name}." for adapter_name in adapter_names)
        for name, param in adapter_model.named_parameters():
            if any(adapter_token in name for adapter_token in adapter_tokens):
                param.requires_grad_(True)

    def _init_lora_mixture_backend(self, peft_config) -> None:
        adapter_config = self._peft_adapter_config(peft_config)
        if adapter_config is None or not hasattr(self.model, "add_adapter"):
            raise ValueError(
                "mixture_reward_backend='lora' requires PEFT/LoRA training. "
                "Pass --use_peft and LoRA target modules."
            )

        self.policy_adapter_name = self._get_active_adapter_name() or "default"
        self.mixture_adapter_names = [
            f"{self.mixture_config.mixture_lora_adapter_prefix}_{cluster_idx}"
            for cluster_idx in range(self.num_clusters)
        ]

        existing_adapters = set(getattr(self.model, "peft_config", {}).keys())
        for adapter_name in self.mixture_adapter_names:
            if adapter_name not in existing_adapters:
                self.model.add_adapter(adapter_name, adapter_config)

        self.mixture_router = MixtureRouterHead(
            hidden_size=self.hidden_size,
            num_clusters=self.num_clusters,
            use_contextual=self.mixture_config.use_contextual_router,
            router_hidden_size=self.mixture_config.router_hidden_size,
        )
        if (
            not self.mixture_config.use_contextual_router
            and getattr(self.mixture_config, "use_closed_form_router_prior_update", False)
            and hasattr(self.mixture_router, "global_logits")
        ):
            self.mixture_router.global_logits.requires_grad_(False)
        self.mixture_router.to(self.args.device)
        self.model.add_module("mixture_router", self.mixture_router)

        self._set_active_adapter(self.policy_adapter_name)
        self._set_trainable_lora_adapters([self.policy_adapter_name, *self.mixture_adapter_names])

    def _update_global_router_prior_from_gamma(self, gamma: torch.Tensor) -> None:
        """Update the non-contextual router logits to the closed-form prior implied by gamma."""
        if self.mixture_config.use_contextual_router:
            return
        if not getattr(self.mixture_config, "use_closed_form_router_prior_update", False):
            return
        router = self.mixture_head.router if self.mixture_reward_backend == "head" else self.mixture_router
        if router is None or not hasattr(router, "global_logits"):
            return

        prior = gamma.mean(dim=0)
        prior = prior / prior.sum().clamp_min(1e-12)
        with torch.no_grad():
            router.global_logits.copy_(torch.log(prior.clamp_min(1e-12)))

    @staticmethod
    def _preference_dimension_sort_key(dimension: str) -> tuple[int, int, str]:
        match = re.fullmatch(r"cluster[_-]?(\d+)", str(dimension))
        if match is not None:
            return (0, int(match.group(1)), str(dimension))
        return (1, 0, str(dimension))

    def _infer_preference_dimension_to_cluster(self, *datasets) -> Dict[str, int]:
        dimensions = set()
        for dataset in datasets:
            if dataset is None:
                continue
            column_names = getattr(dataset, "column_names", [])
            if "preference_dimension" not in column_names:
                continue
            try:
                dimensions.update(str(dimension) for dimension in dataset["preference_dimension"])
            except Exception:
                continue

        sorted_dimensions = sorted(dimensions, key=self._preference_dimension_sort_key)
        return {dimension: idx for idx, dimension in enumerate(sorted_dimensions)}

    @staticmethod
    def _observed_rankings(candidate_mask: torch.Tensor) -> torch.Tensor:
        """Return the dataset-provided listwise order: response 0 is preferred to response 1, etc."""
        batch_size, num_candidates = candidate_mask.shape
        return torch.arange(num_candidates, device=candidate_mask.device).unsqueeze(0).expand(batch_size, -1)

    def _pool_last_hidden_state(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp_min(1.0)

    def _get_batch_mixture_output_head_approx(
        self,
        model,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
        batch: Dict[str, Any],
        batch_size: int,
        num_candidates: int,
    ) -> Dict[str, torch.Tensor]:
        seq_len = flat_input_ids.shape[1]
        candidate_mask = batch["candidate_mask"]
        embedding_layer = model.get_input_embeddings()
        candidate_embeddings = embedding_layer(flat_input_ids).detach().view(batch_size, num_candidates, seq_len, -1)
        anchor_positions = self._select_linear_approx_anchors(candidate_mask)
        anchor_flat_indices = [
            row_idx * num_candidates + position
            for row_idx, positions in enumerate(anchor_positions)
            for position in positions.tolist()
        ]
        if not anchor_flat_indices:
            router_logits = self.mixture_head.router(
                torch.zeros(
                    batch_size,
                    self.hidden_size,
                    device=flat_input_ids.device,
                    dtype=next(self.mixture_head.parameters()).dtype,
                )
            )
            return {
                "router_logits": router_logits,
                "rewards": flat_attention_mask.new_zeros(
                    (batch_size, self.num_clusters, num_candidates),
                    dtype=next(self.mixture_head.parameters()).dtype,
                ),
                "pooled": None,
            }

        anchor_flat_indices_tensor = torch.tensor(anchor_flat_indices, device=flat_input_ids.device, dtype=torch.long)
        anchor_embeds = candidate_embeddings.view(batch_size * num_candidates, seq_len, -1).index_select(
            0, anchor_flat_indices_tensor
        )
        anchor_embeds = anchor_embeds.detach().requires_grad_(True)
        anchor_attention_mask = flat_attention_mask.index_select(0, anchor_flat_indices_tensor)

        mixture_param = next(self.mixture_head.parameters())
        with torch.enable_grad():
            model_output = model(
                inputs_embeds=anchor_embeds,
                attention_mask=anchor_attention_mask,
                output_hidden_states=True,
            )
            pooled = self._pool_last_hidden_state(model_output.hidden_states[-1], anchor_attention_mask)
            pooled_for_head = pooled.to(dtype=mixture_param.dtype)
            anchor_rewards = torch.stack(
                [head(pooled_for_head).squeeze(-1) for head in self.mixture_head.reward_heads],
                dim=1,
            )
            anchor_grads = []
            for cluster_idx in range(self.num_clusters):
                anchor_grads.append(
                    torch.autograd.grad(
                        anchor_rewards[:, cluster_idx].sum(),
                        anchor_embeds,
                        retain_graph=True,
                        create_graph=False,
                    )[0].detach()
                )
            anchor_grads = torch.stack(anchor_grads, dim=1)

        rewards = self._linear_approx_from_anchor_tensors(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_rewards,
            anchor_grads,
        )
        router_context = torch.zeros(
            batch_size,
            self.hidden_size,
            device=flat_input_ids.device,
            dtype=mixture_param.dtype,
        )
        router_logits = self.mixture_head.router(router_context)

        return {
            "router_logits": router_logits,
            "rewards": rewards,
            "pooled": None,
        }

    def _get_batch_mixture_output_head(
        self,
        model,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
        batch: Dict[str, Any],
        batch_size: int,
        num_candidates: int,
    ) -> Dict[str, torch.Tensor]:
        model_output = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            output_hidden_states=True,
        )
        pooled = self._pool_last_hidden_state(model_output.hidden_states[-1], flat_attention_mask)
        pooled = pooled.view(batch_size, num_candidates, self.hidden_size)

        mixture_param = next(self.mixture_head.parameters())
        pooled_for_head = pooled.to(dtype=mixture_param.dtype)
        router_logits, rewards = self.mixture_head(pooled_for_head, batch["candidate_mask"])

        return {
            "router_logits": router_logits,
            "rewards": rewards,
            "pooled": pooled,
        }

    def _get_batch_mixture_output_lora(
        self,
        model,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
        flat_labels: torch.Tensor,
        batch: Dict[str, Any],
        batch_size: int,
        num_candidates: int,
    ) -> Dict[str, torch.Tensor]:
        if self.policy_adapter_name is None or self.mixture_router is None:
            raise RuntimeError("LoRA mixture backend was not initialized.")

        self._set_active_adapter(self.policy_adapter_name)
        if self.mixture_config.use_contextual_router:
            context_output = model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                output_hidden_states=True,
            )
            pooled = self._pool_last_hidden_state(context_output.hidden_states[-1], flat_attention_mask)
            pooled = pooled.view(batch_size, num_candidates, self.hidden_size)
            weights = batch["candidate_mask"].to(dtype=pooled.dtype).unsqueeze(-1)
            context = (pooled * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            router_param = next(self.mixture_router.parameters())
            pooled = None
            context = torch.zeros(
                batch_size,
                self.hidden_size,
                device=flat_input_ids.device,
                dtype=router_param.dtype,
            )

        router_logits = self.mixture_router(context.to(dtype=next(self.mixture_router.parameters()).dtype))
        ref_logps = self._forward_ref(flat_input_ids, flat_attention_mask, flat_labels).view(batch_size, num_candidates)

        beta = self.listwise_beta_override if self.listwise_beta_override is not None else self.beta
        reward_tensors = []
        for adapter_name in self.mixture_adapter_names:
            self._set_active_adapter(adapter_name)
            adapter_logits = model(input_ids=flat_input_ids, attention_mask=flat_attention_mask).logits
            adapter_logps = self._sequence_logps(adapter_logits, flat_labels).view(batch_size, num_candidates)
            reward_tensors.append(beta * (adapter_logps - ref_logps))

        self._set_active_adapter(self.policy_adapter_name)
        self._set_trainable_lora_adapters([self.policy_adapter_name, *self.mixture_adapter_names])

        return {
            "router_logits": router_logits,
            "rewards": torch.stack(reward_tensors, dim=1),
            "pooled": pooled,
        }

    def _get_batch_mixture_output_lora_approx(
        self,
        model,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
        flat_labels: torch.Tensor,
        batch: Dict[str, Any],
        batch_size: int,
        num_candidates: int,
    ) -> Dict[str, torch.Tensor]:
        if self.policy_adapter_name is None or self.mixture_router is None:
            raise RuntimeError("LoRA mixture backend was not initialized.")

        router_param = next(self.mixture_router.parameters())
        context = torch.zeros(
            batch_size,
            self.hidden_size,
            device=flat_input_ids.device,
            dtype=router_param.dtype,
        )
        router_logits = self.mixture_router(context)

        seq_len = flat_input_ids.shape[1]
        candidate_mask = batch["candidate_mask"]
        embedding_layer = model.get_input_embeddings()
        candidate_embeddings = embedding_layer(flat_input_ids).detach().view(batch_size, num_candidates, seq_len, -1)
        anchor_positions = self._select_linear_approx_anchors(candidate_mask)
        anchor_flat_indices = [
            row_idx * num_candidates + position
            for row_idx, positions in enumerate(anchor_positions)
            for position in positions.tolist()
        ]
        if not anchor_flat_indices:
            self._set_active_adapter(self.policy_adapter_name)
            self._set_trainable_lora_adapters([self.policy_adapter_name, *self.mixture_adapter_names])
            return {
                "router_logits": router_logits,
                "rewards": flat_attention_mask.new_zeros(
                    (batch_size, self.num_clusters, num_candidates),
                    dtype=router_param.dtype,
                ),
                "pooled": None,
            }

        anchor_flat_indices_tensor = torch.tensor(anchor_flat_indices, device=flat_input_ids.device, dtype=torch.long)
        anchor_embeds = candidate_embeddings.view(batch_size * num_candidates, seq_len, -1).index_select(
            0, anchor_flat_indices_tensor
        )
        anchor_embeds = anchor_embeds.detach().requires_grad_(True)
        anchor_attention_mask = flat_attention_mask.index_select(0, anchor_flat_indices_tensor)
        anchor_labels = flat_labels.index_select(0, anchor_flat_indices_tensor)
        ref_mode = getattr(self.mixture_config, "linear_approx_ref_mode", "input_gradient")
        beta = self.listwise_beta_override if self.listwise_beta_override is not None else self.beta

        if ref_mode == "exact_score":
            ref_logps = self._forward_ref(flat_input_ids, flat_attention_mask, flat_labels).view(
                batch_size,
                num_candidates,
            )
            ref_grads = None
        else:
            with torch.enable_grad():
                ref_logps = self._forward_ref_from_embeds(anchor_embeds, anchor_attention_mask, anchor_labels)
                ref_grads = torch.autograd.grad(
                    ref_logps.sum(),
                    anchor_embeds,
                    retain_graph=False,
                    create_graph=False,
                )[0].detach()

        reward_tensors = []
        for adapter_name in self.mixture_adapter_names:
            self._set_active_adapter(adapter_name)
            with torch.enable_grad():
                adapter_logits = model(inputs_embeds=anchor_embeds, attention_mask=anchor_attention_mask).logits
                adapter_logps = self._sequence_logps(adapter_logits, anchor_labels)
                adapter_grads = torch.autograd.grad(
                    adapter_logps.sum(),
                    anchor_embeds,
                    retain_graph=True,
                    create_graph=False,
                )[0].detach()
                if ref_mode == "exact_score":
                    anchor_scores = adapter_logps
                    anchor_grads = adapter_grads
                else:
                    anchor_scores = beta * (adapter_logps - ref_logps.detach())
                    anchor_grads = beta * (adapter_grads - ref_grads)

            estimates = self._linear_approx_from_anchor_tensors(
                candidate_embeddings,
                candidate_mask,
                anchor_positions,
                anchor_scores,
                anchor_grads,
            )
            if ref_mode == "exact_score":
                estimates = beta * (estimates - ref_logps.detach())
            reward_tensors.append(estimates)

        self._set_active_adapter(self.policy_adapter_name)
        self._set_trainable_lora_adapters([self.policy_adapter_name, *self.mixture_adapter_names])

        return {
            "router_logits": router_logits,
            "rewards": torch.stack(reward_tensors, dim=1),
            "pooled": None,
        }

    def get_batch_mixture_output(
        self,
        model,
        batch: Dict[str, Any],
        train_eval: Literal["train", "eval"] = "eval",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mixture model outputs: router logits and component rewards.

        Args:
            model: language model
            batch: batch dict with 'input_ids' and 'attention_mask'

        Returns:
            dict with 'router_logits' [B, K] and 'rewards' [B, K, M]
        """
        batch_size, num_candidates, seq_len = batch["input_ids"].shape
        flat_input_ids = batch["input_ids"].view(batch_size * num_candidates, seq_len)
        flat_attention_mask = batch["attention_mask"].view(batch_size * num_candidates, seq_len)
        flat_labels = batch["labels"].view(batch_size * num_candidates, seq_len)

        if self.mixture_reward_backend == "head":
            if self._use_linear_reward_approx(train_eval) and not self.mixture_config.use_contextual_router:
                return self._get_batch_mixture_output_head_approx(
                    model,
                    flat_input_ids,
                    flat_attention_mask,
                    batch,
                    batch_size,
                    num_candidates,
                )
            return self._get_batch_mixture_output_head(
                model,
                flat_input_ids,
                flat_attention_mask,
                batch,
                batch_size,
                num_candidates,
            )

        if self._use_linear_reward_approx(train_eval) and not self.mixture_config.use_contextual_router:
            return self._get_batch_mixture_output_lora_approx(
                model,
                flat_input_ids,
                flat_attention_mask,
                flat_labels,
                batch,
                batch_size,
                num_candidates,
            )

        return self._get_batch_mixture_output_lora(
            model,
            flat_input_ids,
            flat_attention_mask,
            flat_labels,
            batch,
            batch_size,
            num_candidates,
        )

    def _cluster_assignment_metrics(
        self,
        mixture_output: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        assignment_scores: Optional[torch.Tensor] = None,
        metric_prefix: str = "mixture/cluster",
    ) -> Dict[str, float]:
        """Compute raw and permutation-aligned cluster accuracy when labels are available."""
        if not self.mixture_config.log_cluster_metrics or "preference_dimension" not in batch:
            return {}

        true_cluster_values = batch.get("preference_dimension", [])
        cluster_to_dimension: dict[int, str] = {}
        if isinstance(true_cluster_values, torch.Tensor):
            if true_cluster_values.numel() == 0:
                return {}
            true_clusters = true_cluster_values.to(device=mixture_output["router_logits"].device, dtype=torch.long)
            cluster_to_dimension = {
                int(cluster_id): str(int(cluster_id))
                for cluster_id in true_clusters.detach().cpu().unique().tolist()
            }
        else:
            if not true_cluster_values:
                return {}
            if isinstance(true_cluster_values[0], str):
                cluster_matches = [
                    re.fullmatch(r"cluster[_-]?(\d+)", str(dimension)) for dimension in true_cluster_values
                ]
                if all(match is not None for match in cluster_matches):
                    true_cluster_ids = [int(match.group(1)) for match in cluster_matches if match is not None]
                    cluster_to_dimension = {
                        cluster_id: f"cluster_{cluster_id}" for cluster_id in sorted(set(true_cluster_ids))
                    }
                else:
                    configured_dimensions = (
                        getattr(self.args, "preference_dimensions", None)
                        or getattr(self.mixture_config, "preference_dimensions", None)
                        or []
                    )
                    if configured_dimensions:
                        dim_to_cluster = {dim: idx for idx, dim in enumerate(configured_dimensions)}
                    elif self.preference_dimension_to_cluster:
                        dim_to_cluster = self.preference_dimension_to_cluster
                    else:
                        dim_to_cluster = {dim: idx for idx, dim in enumerate(sorted(set(true_cluster_values)))}
                    true_cluster_ids = [dim_to_cluster.get(dim, 0) for dim in true_cluster_values]
                    cluster_to_dimension = {idx: dim for dim, idx in dim_to_cluster.items()}
            else:
                true_cluster_ids = [int(cluster_id) for cluster_id in true_cluster_values]
                cluster_to_dimension = {
                    cluster_id: str(cluster_id) for cluster_id in sorted(set(true_cluster_ids))
                }

            true_clusters = torch.tensor(
                true_cluster_ids,
                device=mixture_output["router_logits"].device,
                dtype=torch.long,
            )

        if assignment_scores is None:
            assignment_scores = mixture_output["router_logits"]
        pred_clusters = assignment_scores.argmax(dim=1).detach()
        true_clusters = true_clusters.detach()
        try:
            pred_clusters = self.accelerator.gather_for_metrics(pred_clusters)
            true_clusters = self.accelerator.gather_for_metrics(true_clusters)
        except Exception:
            pass

        if pred_clusters.numel() == 0 or true_clusters.numel() == 0:
            return {}

        raw_cluster_acc = (pred_clusters == true_clusters).float().mean()

        # Align predicted cluster ids to ground-truth ids using Hungarian matching.
        num_clusters = int(
            max(
                self.num_clusters,
                int(true_clusters.max().item()) + 1,
                int(pred_clusters.max().item()) + 1,
            )
        )
        confusion = torch.zeros((num_clusters, num_clusters), dtype=torch.long, device=pred_clusters.device)
        for true_cluster, pred_cluster in zip(true_clusters.view(-1), pred_clusters.view(-1)):
            confusion[true_cluster.long(), pred_cluster.long()] += 1

        row_ind, col_ind = linear_sum_assignment((-confusion).cpu().numpy())
        pred_to_true = {pred_idx: true_idx for true_idx, pred_idx in zip(row_ind, col_ind)}
        aligned_pred_clusters = torch.tensor(
            [pred_to_true.get(int(pred_cluster.item()), int(pred_cluster.item())) for pred_cluster in pred_clusters],
            device=pred_clusters.device,
        )
        cluster_acc = (aligned_pred_clusters == true_clusters).float().mean()

        cluster_acc_value = float(cluster_acc.item())
        self.router_accs_buffer.append(cluster_acc_value)
        metrics = {
            f"{metric_prefix}_acc": cluster_acc_value,
            f"{metric_prefix}_acc_raw": float(raw_cluster_acc.item()),
        }
        if metric_prefix == "mixture/cluster":
            for cluster_id in range(num_clusters):
                cluster_mask = true_clusters == cluster_id
                num_examples = int(cluster_mask.sum().item())
                if num_examples == 0:
                    continue
                dim_key = self._metric_dimension_key(cluster_to_dimension.get(cluster_id, str(cluster_id)))
                dim_raw_acc = (pred_clusters[cluster_mask] == true_clusters[cluster_mask]).float().mean()
                dim_aligned_acc = (
                    aligned_pred_clusters[cluster_mask] == true_clusters[cluster_mask]
                ).float().mean()
                metrics[f"mixture/by_dimension/{dim_key}/num_examples"] = float(num_examples)
                metrics[f"mixture/by_dimension/{dim_key}/cluster_acc"] = float(dim_aligned_acc.item())
                metrics[f"mixture/by_dimension/{dim_key}/cluster_acc_raw"] = float(dim_raw_acc.item())
        return metrics

    def _mixture_posterior_ranking_metrics(
        self,
        mixture_output: Dict[str, torch.Tensor],
        gamma: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        rewards = mixture_output.get("rewards")
        if rewards is None:
            return {}

        candidate_mask = batch["candidate_mask"]
        ranked_prefix_lengths = batch.get("ranked_prefix_length")
        preference_dimensions = batch.get("preference_dimension")
        pred_clusters = gamma.argmax(dim=1)
        row_indices = torch.arange(rewards.shape[0], device=rewards.device)
        posterior_rewards = rewards[row_indices, pred_clusters, :]

        metric_tensors = self._listwise_metrics(posterior_rewards, candidate_mask, ranked_prefix_lengths)
        metrics = {
            f"mixture_posterior/{name.removeprefix('listwise/')}": self.accelerator.gather_for_metrics(
                value.detach()
            ).mean().item()
            for name, value in metric_tensors.items()
            if name in {"listwise/top1_acc", "listwise/pairwise_acc"}
        }

        if preference_dimensions is not None:
            dim_to_indices: dict[str, list[int]] = {}
            for idx, dimension in enumerate(preference_dimensions):
                dim_key = self._metric_dimension_key(dimension)
                dim_to_indices.setdefault(dim_key, []).append(idx)

            for dim_key, indices in dim_to_indices.items():
                dim_index_tensor = torch.tensor(indices, device=posterior_rewards.device, dtype=torch.long)
                dim_metrics = self._listwise_metrics(
                    posterior_rewards.index_select(0, dim_index_tensor),
                    candidate_mask.index_select(0, dim_index_tensor),
                    ranked_prefix_lengths.index_select(0, dim_index_tensor)
                    if ranked_prefix_lengths is not None
                    else None,
                )
                for metric_name, metric_value in dim_metrics.items():
                    if metric_name not in {"listwise/top1_acc", "listwise/pairwise_acc"}:
                        continue
                    suffix = metric_name.removeprefix("listwise/")
                    metrics[f"mixture_posterior/by_dimension/{dim_key}/{suffix}"] = self.accelerator.gather_for_metrics(
                        metric_value.detach()
                    ).mean().item()

        return metrics

    def get_batch_mixture_em_loss_metrics(
        self,
        model,
        batch: Dict[str, Any],
        gamma: Optional[torch.Tensor] = None,
        include_cluster_metrics: bool = True,
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor], torch.Tensor]:
        mixture_output = self.get_batch_mixture_output(model, batch, train_eval=train_eval)
        candidate_mask = batch["candidate_mask"]
        ranked_prefix_lengths = batch.get("ranked_prefix_length")
        rankings = self._observed_rankings(candidate_mask).long()
        if ranked_prefix_lengths is None:
            ranked_prefix_lengths = candidate_mask.long().sum(dim=1)
        valid_rows = (candidate_mask.sum(dim=1) >= 2) & (ranked_prefix_lengths >= 1)

        mixture_nll, comp_logp = mixture_pl_nll(
            mixture_output["router_logits"],
            mixture_output["rewards"],
            rankings,
            candidate_mask=candidate_mask,
            ranked_prefix_lengths=ranked_prefix_lengths,
        )
        if gamma is None:
            gamma = em_responsibilities(
                mixture_output["router_logits"],
                comp_logp,
                temperature=self.mixture_config.em_temperature,
            ).detach()

        mixture_em_nll = em_expected_complete_nll(
            mixture_output["router_logits"],
            comp_logp,
            gamma,
            valid_mask=valid_rows,
        )

        metrics = {
            "mixture/nll": float(mixture_nll.detach().item()),
            "mixture/em_nll": float(mixture_em_nll.detach().item()),
        }
        if include_cluster_metrics:
            metrics.update(
                self._cluster_assignment_metrics(
                    mixture_output,
                    batch,
                    assignment_scores=gamma,
                    metric_prefix="mixture/cluster",
                )
            )
            metrics.update(
                self._cluster_assignment_metrics(
                    mixture_output,
                    batch,
                    assignment_scores=mixture_output["router_logits"],
                    metric_prefix="mixture/router_cluster",
                )
            )
            metrics.update(self._mixture_posterior_ranking_metrics(mixture_output, gamma, batch))
        return mixture_em_nll, metrics, mixture_output, gamma

    def _lora_router_em_loss(
        self,
        model,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
        batch_size: int,
        num_candidates: int,
        gamma: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> torch.Tensor:
        if self.mixture_router is None or self.policy_adapter_name is None:
            raise RuntimeError("LoRA mixture backend was not initialized.")

        if self.mixture_config.use_contextual_router:
            self._set_active_adapter(self.policy_adapter_name)
            context_output = model(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
                output_hidden_states=True,
            )
            pooled = self._pool_last_hidden_state(context_output.hidden_states[-1], flat_attention_mask)
            pooled = pooled.view(batch_size, num_candidates, self.hidden_size)
            weights = batch["candidate_mask"].to(dtype=pooled.dtype).unsqueeze(-1)
            context = (pooled * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            router_param = next(self.mixture_router.parameters())
            context = torch.zeros(
                batch_size,
                self.hidden_size,
                device=flat_input_ids.device,
                dtype=router_param.dtype,
            )

        router_logits = self.mixture_router(context.to(dtype=next(self.mixture_router.parameters()).dtype))
        log_alpha = F.log_softmax(router_logits, dim=1)
        row_losses = -(gamma.to(device=log_alpha.device, dtype=log_alpha.dtype) * log_alpha).sum(dim=1)
        if not torch.any(valid_rows):
            return row_losses.sum() * 0.0
        return row_losses[valid_rows].mean()

    def _backward_lora_mixture_em_loss_sequential(
        self,
        model,
        batch: Dict[str, Any],
        gamma: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Backward the LoRA-mixture EM M-step one component at a time.

        With fixed E-step responsibilities, the complete-data objective decomposes
        across mixture components. Backwarding each adapter loss separately avoids
        holding all cluster adapter autograd graphs in memory at once.
        """
        if self.policy_adapter_name is None or self.mixture_router is None:
            raise RuntimeError("LoRA mixture backend was not initialized.")

        batch_size, num_candidates, seq_len = batch["input_ids"].shape
        flat_input_ids = batch["input_ids"].view(batch_size * num_candidates, seq_len)
        flat_attention_mask = batch["attention_mask"].view(batch_size * num_candidates, seq_len)
        flat_labels = batch["labels"].view(batch_size * num_candidates, seq_len)
        candidate_mask = batch["candidate_mask"]
        ranked_prefix_lengths = batch.get("ranked_prefix_length")
        rankings = self._observed_rankings(candidate_mask).long()
        if ranked_prefix_lengths is None:
            ranked_prefix_lengths = candidate_mask.long().sum(dim=1)
        valid_rows = (candidate_mask.sum(dim=1) >= 2) & (ranked_prefix_lengths >= 1)
        gamma = gamma.to(device=flat_input_ids.device)

        component_loss_value = 0.0
        total_loss_value = 0.0
        closed_form_router = (
            not self.mixture_config.use_contextual_router
            and getattr(self.mixture_config, "use_closed_form_router_prior_update", False)
        )

        router_context = torch.no_grad() if closed_form_router else nullcontext()
        with router_context:
            router_loss = self._lora_router_em_loss(
                model,
                flat_input_ids,
                flat_attention_mask,
                batch_size,
                num_candidates,
                gamma,
                valid_rows,
            )
        router_loss_value = float(router_loss.detach().item())
        total_loss_value += router_loss_value
        if not closed_form_router:
            self.accelerator.backward(router_loss)

        beta = self.listwise_beta_override if self.listwise_beta_override is not None else self.beta
        use_linear_m_step = self._use_linear_reward_approx("train") and not self.mixture_config.use_contextual_router

        if use_linear_m_step:
            embedding_layer = model.get_input_embeddings()
            candidate_embeddings = embedding_layer(flat_input_ids).detach().view(
                batch_size,
                num_candidates,
                seq_len,
                -1,
            )
            anchor_positions = self._select_linear_approx_anchors(candidate_mask)
            anchor_flat_indices = [
                row_idx * num_candidates + position
                for row_idx, positions in enumerate(anchor_positions)
                for position in positions.tolist()
            ]

            if anchor_flat_indices:
                anchor_flat_indices_tensor = torch.tensor(
                    anchor_flat_indices,
                    device=flat_input_ids.device,
                    dtype=torch.long,
                )
                anchor_embeds = candidate_embeddings.view(batch_size * num_candidates, seq_len, -1).index_select(
                    0,
                    anchor_flat_indices_tensor,
                )
                anchor_embeds = anchor_embeds.detach().requires_grad_(True)
                anchor_attention_mask = flat_attention_mask.index_select(0, anchor_flat_indices_tensor)
                anchor_labels = flat_labels.index_select(0, anchor_flat_indices_tensor)

                ref_mode = getattr(self.mixture_config, "linear_approx_ref_mode", "input_gradient")
                if ref_mode == "exact_score":
                    ref_logps = self._forward_ref(flat_input_ids, flat_attention_mask, flat_labels).view(
                        batch_size,
                        num_candidates,
                    )
                    ref_grads = None
                else:
                    with torch.enable_grad():
                        ref_logps = self._forward_ref_from_embeds(anchor_embeds, anchor_attention_mask, anchor_labels)
                        ref_grads = torch.autograd.grad(
                            ref_logps.sum(),
                            anchor_embeds,
                            retain_graph=False,
                            create_graph=False,
                        )[0].detach()

                for cluster_idx, adapter_name in enumerate(self.mixture_adapter_names):
                    self._set_active_adapter(adapter_name)
                    with torch.enable_grad():
                        adapter_logits = model(inputs_embeds=anchor_embeds, attention_mask=anchor_attention_mask).logits
                        adapter_logps = self._sequence_logps(adapter_logits, anchor_labels)
                        adapter_grads = torch.autograd.grad(
                            adapter_logps.sum(),
                            anchor_embeds,
                            retain_graph=True,
                            create_graph=False,
                        )[0].detach()
                        if ref_mode == "exact_score":
                            anchor_scores = adapter_logps
                            anchor_grads = adapter_grads
                        else:
                            anchor_scores = beta * (adapter_logps - ref_logps.detach())
                            anchor_grads = beta * (adapter_grads - ref_grads)

                    estimates = self._linear_approx_from_anchor_tensors(
                        candidate_embeddings,
                        candidate_mask,
                        anchor_positions,
                        anchor_scores,
                        anchor_grads,
                    )
                    rewards = beta * (estimates - ref_logps.detach()) if ref_mode == "exact_score" else estimates
                    comp_logp = pl_log_prob(
                        rewards,
                        rankings,
                        candidate_mask=candidate_mask,
                        ranked_prefix_lengths=ranked_prefix_lengths,
                    )
                    weighted_comp_nll = -(gamma[:, cluster_idx].to(dtype=comp_logp.dtype) * comp_logp)
                    if torch.any(valid_rows):
                        component_loss = weighted_comp_nll[valid_rows].mean()
                    else:
                        component_loss = weighted_comp_nll.sum() * 0.0

                    self.accelerator.backward(component_loss)
                    if anchor_embeds.grad is not None:
                        anchor_embeds.grad = None
                    loss_value = float(component_loss.detach().item())
                    component_loss_value += loss_value
                    total_loss_value += loss_value

                    del (
                        adapter_logits,
                        adapter_logps,
                        adapter_grads,
                        anchor_scores,
                        anchor_grads,
                        estimates,
                        rewards,
                        comp_logp,
                        weighted_comp_nll,
                        component_loss,
                    )

                del anchor_embeds, anchor_attention_mask, anchor_labels
                if ref_mode == "exact_score":
                    del ref_logps
                else:
                    del ref_logps, ref_grads
        else:
            ref_logps = self._forward_ref(flat_input_ids, flat_attention_mask, flat_labels).view(batch_size, num_candidates)

            for cluster_idx, adapter_name in enumerate(self.mixture_adapter_names):
                self._set_active_adapter(adapter_name)
                adapter_logits = model(input_ids=flat_input_ids, attention_mask=flat_attention_mask).logits
                adapter_logps = self._sequence_logps(adapter_logits, flat_labels).view(batch_size, num_candidates)
                rewards = beta * (adapter_logps - ref_logps)
                comp_logp = pl_log_prob(
                    rewards,
                    rankings,
                    candidate_mask=candidate_mask,
                    ranked_prefix_lengths=ranked_prefix_lengths,
                )
                weighted_comp_nll = -(gamma[:, cluster_idx].to(dtype=comp_logp.dtype) * comp_logp)
                if torch.any(valid_rows):
                    component_loss = weighted_comp_nll[valid_rows].mean()
                else:
                    component_loss = weighted_comp_nll.sum() * 0.0

                self.accelerator.backward(component_loss)
                loss_value = float(component_loss.detach().item())
                component_loss_value += loss_value
                total_loss_value += loss_value

                del adapter_logits, adapter_logps, rewards, comp_logp, weighted_comp_nll, component_loss
            del ref_logps

        self._set_active_adapter(self.policy_adapter_name)
        self._set_trainable_lora_adapters([self.policy_adapter_name, *self.mixture_adapter_names])

        metrics = {
            "mixture/em_nll": total_loss_value,
            "mixture/router_em_nll": router_loss_value,
            "mixture/component_em_nll": component_loss_value,
        }
        return total_loss_value, metrics

    def get_batch_loss_metrics_with_mixture(
        self,
        model,
        batch: Dict[str, Any],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
        """
        Compute DPO loss + mixture loss and all metrics.

        Returns:
            loss: hybrid loss (DPO + mixture)
            metrics: dict of computed metrics
            mixture_output: intermediate mixture outputs (for M-step caching)
        """
        if self.mixture_reward_backend == "lora":
            self._set_active_adapter(self.policy_adapter_name)
            self._set_trainable_lora_adapters([self.policy_adapter_name, *self.mixture_adapter_names])

        # ===== Compute DPO loss (listwise) =====
        dpo_loss, dpo_metrics = self.get_batch_loss_metrics(model, batch, train_eval=train_eval)

        # ===== Compute mixture loss (if applicable) =====
        mixture_metrics = {}
        mixture_output = None

        if train_eval == "train" or self.mixture_config.log_cluster_metrics:
            mixture_em_nll, mixture_metrics, mixture_output, gamma = self.get_batch_mixture_em_loss_metrics(
                model,
                batch,
                train_eval=train_eval,
            )

            if train_eval == "train":
                self._update_global_router_prior_from_gamma(gamma)
                if self.mixture_config.mixture_training_mode == "em_only":
                    hybrid_loss = mixture_em_nll
                else:
                    hybrid_loss = dpo_loss + self.mixture_config.mixture_nll_weight * mixture_em_nll
            else:
                hybrid_loss = dpo_loss

        else:
            hybrid_loss = dpo_loss

        # Combine metrics
        all_metrics = {**dpo_metrics, **mixture_metrics}

        return hybrid_loss, all_metrics, mixture_output

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to add mixture loss to DPO loss."""
        del kwargs
        compute_loss_context_manager = (
            _peft_bf16_autocast_context(self)
        )
        with compute_loss_context_manager:
            loss, metrics, _ = self.get_batch_loss_metrics_with_mixture(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        """Override to include mixture metrics in evaluation."""
        del ignore_keys
        prediction_context_manager = (
            _peft_bf16_autocast_context(self)
        )
        with torch.no_grad(), prediction_context_manager:
            loss, metrics, _ = self.get_batch_loss_metrics_with_mixture(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = torch.tensor(
            [
                metrics.get("eval_listwise/utility_first", 0.0),
                metrics.get("eval_listwise/utility_last", 0.0),
            ],
            device=self.accelerator.device,
        )
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            if self.mixture_reward_backend == "head":
                torch.save(self.mixture_head.state_dict(), os.path.join(output_dir, "mixture_pl_head.pt"))
            else:
                torch.save(self.mixture_router.state_dict(), os.path.join(output_dir, "mixture_router.pt"))

    def _load_mixture_state_from_checkpoint(self, checkpoint_dir: str) -> None:
        if self.mixture_reward_backend == "head":
            mixture_head_path = os.path.join(checkpoint_dir, "mixture_pl_head.pt")
            if os.path.exists(mixture_head_path):
                state_dict = torch.load(mixture_head_path, map_location=self.args.device)
                self.mixture_head.load_state_dict(state_dict)
        else:
            mixture_router_path = os.path.join(checkpoint_dir, "mixture_router.pt")
            if os.path.exists(mixture_router_path):
                state_dict = torch.load(mixture_router_path, map_location=self.args.device)
                self.mixture_router.load_state_dict(state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._load_mixture_state_from_checkpoint(resume_from_checkpoint)
        return result


class MixtureEMDPOTrainer(MixtureDPOTrainer):
    """Mixture trainer with a classic EM-style loop over the mixture objective only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mixture_config.mixture_training_mode != "em_only":
            raise ValueError("MixtureEMDPOTrainer requires mixture_training_mode='em_only'")

    @staticmethod
    def _strategy_value(strategy) -> str:
        return str(getattr(strategy, "value", strategy)).lower()

    @classmethod
    def _strategy_is(cls, strategy, expected: str) -> bool:
        value = cls._strategy_value(strategy)
        return value == expected or value.endswith(f".{expected}")

    @staticmethod
    def _positive_interval(value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _release_cuda_cache_after_eval_or_save(self) -> None:
        """Release cached CUDA blocks after memory-heavy eval/checkpoint phases."""
        if not torch.cuda.is_available():
            return
        self.accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()

    def _update_best_metric_for_legacy_checkpointing(self, metrics: Optional[dict[str, float]]) -> None:
        """Update TrainerState best fields for Trainer versions whose _save_checkpoint lacks metrics=."""
        if not metrics:
            return
        metric_name = getattr(self.args, "metric_for_best_model", None)
        if metric_name is None:
            return

        metric_value = metrics.get(metric_name)
        if metric_value is None and not str(metric_name).startswith("eval_"):
            metric_value = metrics.get(f"eval_{metric_name}")
        if metric_value is None:
            return

        operator = float.__gt__ if getattr(self.args, "greater_is_better", False) else float.__lt__
        current_best = self.state.best_metric
        if current_best is None or operator(float(metric_value), float(current_best)):
            self.state.best_metric = float(metric_value)
            self.state.best_global_step = self.state.global_step

    def _maybe_evaluate_and_save_em(self, trial=None, force_epoch: bool = False):
        args = self.args
        eval_strategy = getattr(args, "eval_strategy", getattr(args, "evaluation_strategy", "no"))
        save_strategy = getattr(args, "save_strategy", "no")
        eval_metrics = None

        should_eval = False
        if force_epoch:
            should_eval = self._strategy_is(eval_strategy, "epoch")
        elif self._strategy_is(eval_strategy, "steps"):
            eval_steps = self._positive_interval(getattr(args, "eval_steps", 0))
            should_eval = eval_steps > 0 and self.state.global_step % eval_steps == 0

        if should_eval and self.eval_dataset is not None:
            eval_metrics = self.evaluate()
            self._release_cuda_cache_after_eval_or_save()

        should_save = False
        if force_epoch:
            should_save = self._strategy_is(save_strategy, "epoch")
        elif self._strategy_is(save_strategy, "steps"):
            save_steps = self._positive_interval(getattr(args, "save_steps", 0))
            should_save = save_steps > 0 and self.state.global_step % save_steps == 0

        if should_save:
            if (
                eval_metrics is None
                and getattr(args, "metric_for_best_model", None) is not None
                and self.eval_dataset is not None
            ):
                eval_metrics = self.evaluate()
                self._release_cuda_cache_after_eval_or_save()
            try:
                self._save_checkpoint(self.model, trial=trial, metrics=eval_metrics)
            except TypeError as exc:
                if "unexpected keyword argument 'metrics'" not in str(exc):
                    raise
                self._update_best_metric_for_legacy_checkpointing(eval_metrics)
                self._save_checkpoint(self.model, trial=trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            self._release_cuda_cache_after_eval_or_save()

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):  # noqa: D401
        del ignore_keys_for_eval, kwargs
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)

        args = self.args
        train_dataloader = self.get_train_dataloader()
        m_step_updates = self.mixture_config.m_step_updates
        steps_per_epoch = max(len(train_dataloader) * m_step_updates, 1)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = math.ceil(max_steps / steps_per_epoch)
        else:
            num_train_epochs = math.ceil(args.num_train_epochs)
            max_steps = math.ceil(args.num_train_epochs * steps_per_epoch)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.model, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
            self.lr_scheduler,
        )
        self.model_wrapped = self.model
        self.optimizer.zero_grad()

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.global_step = 0

        total_loss = 0.0
        start_time = time.time()
        stop_training = False

        for epoch in range(num_train_epochs):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                inputs = self._prepare_inputs(batch)

                # E-step: infer gamma once with current parameters.
                with torch.no_grad():
                    _, _, _, gamma = self.get_batch_mixture_em_loss_metrics(
                        self.model,
                        inputs,
                        include_cluster_metrics=False,
                    )
                self._update_global_router_prior_from_gamma(gamma)

                # M-step: optimize only E_q[log p(z, ranking | x)] for fixed gamma.
                for m_step in range(m_step_updates):
                    self.model.train()
                    should_log = args.logging_steps > 0 and (self.state.global_step + 1) % args.logging_steps == 0
                    if self.mixture_reward_backend == "lora":
                        loss_value, metrics = self._backward_lora_mixture_em_loss_sequential(
                            self.model,
                            inputs,
                            gamma=gamma,
                        )
                    else:
                        loss, metrics, _, _ = self.get_batch_mixture_em_loss_metrics(
                            self.model,
                            inputs,
                            gamma=gamma,
                        )
                    metric_context_manager = (
                        torch.autocast(self.accelerator.device.type)
                        if self._peft_has_been_casted_to_bf16
                        else nullcontext()
                    )
                    with torch.no_grad(), metric_context_manager:
                        if self.mixture_reward_backend == "lora":
                            self._set_active_adapter(self.policy_adapter_name)
                        _, listwise_metrics = self.get_batch_loss_metrics(
                            self.model,
                            inputs,
                            train_eval="train",
                        )
                    metrics.update(listwise_metrics)
                    self.store_metrics(metrics, train_eval="train")

                    if self.mixture_reward_backend != "lora":
                        self.accelerator.backward(loss)
                        loss_value = float(loss.detach().item())
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.state.global_step += 1
                    total_loss += loss_value

                    if should_log:
                        self.log(
                            {
                                "loss": loss_value,
                                "em/e_step_epoch": float(epoch),
                                "em/m_step": float(m_step + 1),
                            }
                        )

                    self._maybe_evaluate_and_save_em(trial=trial)

                    if self.state.global_step >= max_steps:
                        stop_training = True
                        break

                if stop_training:
                    break

            if not stop_training:
                self.state.epoch = float(epoch + 1)
                self._maybe_evaluate_and_save_em(trial=trial, force_epoch=True)

            if stop_training:
                break

        if getattr(args, "load_best_model_at_end", False) and self.state.best_model_checkpoint is not None:
            self._load_best_model()
            self._load_mixture_state_from_checkpoint(self.state.best_model_checkpoint)

        runtime = time.time() - start_time
        train_loss = total_loss / max(self.state.global_step, 1)
        metrics = {
            "train_runtime": runtime,
            "train_samples_per_second": self.state.global_step * args.train_batch_size / max(runtime, 1e-12),
            "train_steps_per_second": self.state.global_step / max(runtime, 1e-12),
            "train_loss": train_loss,
        }
        return TrainOutput(self.state.global_step, train_loss, metrics)
