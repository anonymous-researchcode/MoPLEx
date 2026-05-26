from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import re
from typing import Any, Dict, Literal, Optional, Union

from accelerate import PartialState
from datasets import Dataset, IterableDataset
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from transformers import BaseImageProcessor, FeatureExtractionMixin, PreTrainedTokenizerBase, ProcessorMixin
from trl import DPOTrainer

from .mixture_pl_components import MixturePLHead


@dataclass
class PairwiseBTDataCollator:
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
        batch_preference_dimensions = []

        for feature in features:
            prompt = str(feature["prompt"])
            chosen = str(feature["chosen"])
            rejected = str(feature["rejected"])
            batch_preference_dimensions.append(feature.get("preference_dimension", "unknown"))

            item_input_ids = []
            item_attention_mask = []
            item_labels = []
            for response in (chosen, rejected):
                input_ids, attention_mask, labels = self._build_sequence(prompt, response)
                item_input_ids.append(input_ids)
                item_attention_mask.append(attention_mask)
                item_labels.append(labels)

            batch_input_ids.append(item_input_ids)
            batch_attention_mask.append(item_attention_mask)
            batch_labels.append(item_labels)

        max_seq_len = max(len(seq) for item in batch_input_ids for seq in item)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for pairwise BT collation.")

        for i in range(len(batch_input_ids)):
            for j in range(2):
                seq_len = len(batch_input_ids[i][j])
                pad_len = max_seq_len - seq_len
                batch_input_ids[i][j] = batch_input_ids[i][j] + [pad_id] * pad_len
                batch_attention_mask[i][j] = batch_attention_mask[i][j] + [0] * pad_len
                batch_labels[i][j] = batch_labels[i][j] + [-100] * pad_len

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "preference_dimension": batch_preference_dimensions,
        }


class MixtureBTTrainer(DPOTrainer):
    """Pairwise mixture Bradley-Terry trainer for PERSONA-style comparisons."""

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
        del model_init
        if tokenizer is not None and kwargs.get("processing_class") is None:
            kwargs["processing_class"] = tokenizer
        if kwargs.get("data_collator") is None and data_collator is None:
            processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
            if processing_class is None:
                raise ValueError("A tokenizer/processing_class is required for pairwise BT data collation.")
            max_length = getattr(args, "max_length", 1024) if args is not None else 1024
            max_prompt_length = getattr(args, "max_prompt_length", 512) if args is not None else 512
            data_collator = PairwiseBTDataCollator(
                tokenizer=processing_class,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
            )

        super().__init__(
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

        self.mixture_config = mixture_config
        if mixture_config is None or not mixture_config.use_mixture:
            raise ValueError("MixtureBTTrainer requires mixture_config with use_mixture=True")

        self.num_clusters = mixture_config.num_clusters
        if self.num_clusters is None:
            if hasattr(args, "preference_dimensions") and args.preference_dimensions:
                self.num_clusters = len(args.preference_dimensions)
            else:
                raise ValueError("num_clusters must be specified for mixture BT when preference_dimensions is unset")

        model_config = getattr(self.model, "config", getattr(model, "config", None))
        if model_config is None or not hasattr(model_config, "hidden_size"):
            raise ValueError("Model config must have hidden_size attribute")
        self.hidden_size = model_config.hidden_size

        self.mixture_bt_head = MixturePLHead(
            hidden_size=self.hidden_size,
            num_clusters=self.num_clusters,
            use_contextual_router=mixture_config.use_contextual_router,
            router_hidden_size=mixture_config.router_hidden_size,
        )
        if (
            not mixture_config.use_contextual_router
            and getattr(mixture_config, "use_closed_form_router_prior_update", False)
            and hasattr(self.mixture_bt_head.router, "global_logits")
        ):
            self.mixture_bt_head.router.global_logits.requires_grad_(False)
        self.mixture_bt_head.to(self.args.device)
        self.model.add_module("mixture_bt_head", self.mixture_bt_head)
        self.preference_dimension_to_cluster = self._infer_preference_dimension_to_cluster(
            self.train_dataset,
            self.eval_dataset,
        )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "labels",
                "preference_dimension",
            ]

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: Any,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
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
                fn_kwargs={"tokenizer": processing_class, "tools": args.tools},
                **map_kwargs,
            )
        return dataset

    @staticmethod
    def _preference_dimension_sort_key(dimension: str) -> tuple[int, int, str]:
        match = re.fullmatch(r"cluster[_-]?(\d+)", str(dimension))
        if match is not None:
            return (0, int(match.group(1)), str(dimension))
        match = re.fullmatch(r"persona[_-]?(\d+)", str(dimension))
        if match is not None:
            return (0, int(match.group(1)), str(dimension))
        return (1, 0, str(dimension))

    def _infer_preference_dimension_to_cluster(self, *datasets) -> Dict[str, int]:
        dimensions = set()
        for dataset in datasets:
            if dataset is None:
                continue
            if "preference_dimension" not in getattr(dataset, "column_names", []):
                continue
            try:
                dimensions.update(str(dimension) for dimension in dataset["preference_dimension"])
            except Exception:
                continue
        sorted_dimensions = sorted(dimensions, key=self._preference_dimension_sort_key)
        return {dimension: idx for idx, dimension in enumerate(sorted_dimensions)}

    @staticmethod
    def _pool_last_hidden_state(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp_min(1.0)

    def get_batch_mixture_bt_output(self, model, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        batch_size, num_candidates, seq_len = batch["input_ids"].shape
        flat_input_ids = batch["input_ids"].view(batch_size * num_candidates, seq_len)
        flat_attention_mask = batch["attention_mask"].view(batch_size * num_candidates, seq_len)

        model_output = model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            output_hidden_states=True,
        )
        pooled = self._pool_last_hidden_state(model_output.hidden_states[-1], flat_attention_mask)
        pooled = pooled.view(batch_size, num_candidates, self.hidden_size)
        head_param = next(self.mixture_bt_head.parameters())
        router_logits, rewards = self.mixture_bt_head(
            pooled.to(dtype=head_param.dtype),
            torch.ones((batch_size, num_candidates), dtype=torch.bool, device=pooled.device),
        )
        return {
            "router_logits": router_logits,
            "rewards": rewards,
            "pooled": pooled,
        }

    def _update_global_router_prior_from_gamma(self, gamma: torch.Tensor) -> None:
        if self.mixture_config.use_contextual_router:
            return
        if not getattr(self.mixture_config, "use_closed_form_router_prior_update", False):
            return
        router = self.mixture_bt_head.router
        if router is None or not hasattr(router, "global_logits"):
            return
        prior = gamma.mean(dim=0)
        prior = prior / prior.sum().clamp_min(1e-12)
        with torch.no_grad():
            router.global_logits.copy_(torch.log(prior.clamp_min(1e-12)))

    def _cluster_assignment_metrics(
        self,
        assignment_scores: torch.Tensor,
        batch: Dict[str, Any],
        metric_prefix: str,
    ) -> Dict[str, float]:
        if not self.mixture_config.log_cluster_metrics or "preference_dimension" not in batch:
            return {}
        true_values = batch.get("preference_dimension", [])
        if not true_values:
            return {}
        if isinstance(true_values[0], str):
            dim_to_cluster = self.preference_dimension_to_cluster
            true_ids = [dim_to_cluster.get(str(value), 0) for value in true_values]
        else:
            true_ids = [int(value) for value in true_values]

        true_clusters = torch.tensor(true_ids, dtype=torch.long, device=assignment_scores.device)
        pred_clusters = assignment_scores.argmax(dim=1).detach()
        try:
            pred_clusters = self.accelerator.gather_for_metrics(pred_clusters)
            true_clusters = self.accelerator.gather_for_metrics(true_clusters)
        except Exception:
            pass
        if pred_clusters.numel() == 0:
            return {}

        raw_acc = (pred_clusters == true_clusters).float().mean()
        num_clusters = int(max(self.num_clusters, int(true_clusters.max().item()) + 1, int(pred_clusters.max().item()) + 1))
        confusion = torch.zeros((num_clusters, num_clusters), dtype=torch.long, device=pred_clusters.device)
        for true_cluster, pred_cluster in zip(true_clusters.view(-1), pred_clusters.view(-1)):
            confusion[true_cluster.long(), pred_cluster.long()] += 1
        row_ind, col_ind = linear_sum_assignment(-confusion.detach().cpu().numpy())
        pred_to_true = {int(pred): int(true) for true, pred in zip(row_ind, col_ind)}
        aligned_pred = torch.tensor(
            [pred_to_true.get(int(pred.item()), int(pred.item())) for pred in pred_clusters],
            dtype=torch.long,
            device=pred_clusters.device,
        )
        aligned_acc = (aligned_pred == true_clusters).float().mean()
        return {
            f"{metric_prefix}_acc": float(aligned_acc.item()),
            f"{metric_prefix}_acc_raw": float(raw_acc.item()),
        }

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Any],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        mixture_output = self.get_batch_mixture_bt_output(model, batch)
        router_logits = mixture_output["router_logits"]
        rewards = mixture_output["rewards"]
        reward_diff = rewards[:, :, 0] - rewards[:, :, 1]
        comp_logp = F.logsigmoid(reward_diff)
        log_alpha = F.log_softmax(router_logits, dim=1)
        joint = log_alpha + comp_logp
        mix_logp = torch.logsumexp(joint, dim=1)
        mixture_nll = -mix_logp.mean()
        gamma = F.softmax(joint / self.mixture_config.em_temperature, dim=1).detach()
        mixture_em_nll = -(gamma * joint).sum(dim=1).mean()

        if train_eval == "train":
            self._update_global_router_prior_from_gamma(gamma)
            if self.mixture_config.mixture_training_mode == "em_only":
                loss = mixture_em_nll
            else:
                loss = mixture_nll
        else:
            loss = mixture_nll

        posterior_cluster = gamma.argmax(dim=1)
        row_indices = torch.arange(rewards.shape[0], device=rewards.device)
        posterior_diff = reward_diff[row_indices, posterior_cluster]
        posterior_acc = (posterior_diff > 0).float().mean()
        mix_prob = torch.exp(mix_logp).clamp(0.0, 1.0)

        metric_tensors = {
            "mixture_bt/nll": mixture_nll.detach(),
            "mixture_bt/em_nll": mixture_em_nll.detach(),
            "mixture_bt/posterior_pairwise_acc": posterior_acc.detach(),
            "mixture_bt/mix_prob_chosen_mean": mix_prob.mean().detach(),
        }
        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}{name}": self.accelerator.gather_for_metrics(value).mean().item()
            for name, value in metric_tensors.items()
        }
        if train_eval == "train" or self.mixture_config.log_cluster_metrics:
            cluster_metrics = self._cluster_assignment_metrics(
                gamma,
                batch,
                metric_prefix=f"{prefix}mixture_bt/cluster",
            )
            router_metrics = self._cluster_assignment_metrics(
                router_logits,
                batch,
                metric_prefix=f"{prefix}mixture_bt/router_cluster",
            )
            metrics.update(cluster_metrics)
            metrics.update(router_metrics)
        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        compute_loss_context_manager = (
            torch.autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
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
        ignore_keys: Optional[list[str]] = None,
    ):
        del ignore_keys
        prediction_context_manager = (
            torch.autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")
        self.store_metrics(metrics, train_eval="eval")
        if prediction_loss_only:
            return (loss.detach(), None, None)
        logits = torch.tensor(
            [
                metrics.get("eval_mixture_bt/mix_prob_chosen_mean", 0.0),
                metrics.get("eval_mixture_bt/posterior_pairwise_acc", 0.0),
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
            torch.save(self.mixture_bt_head.state_dict(), os.path.join(output_dir, "mixture_bt_head.pt"))

    def _load_mixture_bt_state_from_checkpoint(self, checkpoint_dir: str) -> None:
        mixture_bt_path = os.path.join(checkpoint_dir, "mixture_bt_head.pt")
        if os.path.exists(mixture_bt_path):
            state_dict = torch.load(mixture_bt_path, map_location=self.args.device)
            self.mixture_bt_head.load_state_dict(state_dict)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._load_mixture_bt_state_from_checkpoint(resume_from_checkpoint)
        return result
