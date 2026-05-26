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

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, Optional

import trl


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )
    dataset_format: str = field(
        default="pairwise",
        metadata={"help": "Dataset format to load: 'pairwise' (default) or 'listwise'."},
    )
    preference_dimensions: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "Ordered preference dimensions for listwise training/conversion. "
                "For non-cyclic listwise, one listwise row is emitted per dimension. "
                "For cyclic listwise, dimensions define rotation checks. "
                "Example: ['instruction_following', 'helpfulness']."
            )
        },
    )
    listwise_num_responses: int = field(
        default=4,
        metadata={"help": "Number of ranked responses per prompt to keep for listwise training."},
    )
    listwise_num_generated_responses: int = field(
        default=0,
        metadata={
            "help": (
                "Number of generated tail responses to append when reducing augmented preformatted listwise data. "
                "When >0, listwise_num_responses is interpreted as the number of original ranked-prefix "
                "responses and this value is the number of generated bottom candidates."
            )
        },
    )
    listwise_prompt_column: str = field(
        default="instruction",
        metadata={"help": "Prompt column in the raw listwise dataset."},
    )
    listwise_responses_column: str = field(
        default="completions",
        metadata={"help": "Column containing candidate responses in listwise datasets."},
    )
    listwise_response_text_key: str = field(
        default="response",
        metadata={"help": "Key used to read response text from each completion item."},
    )
    listwise_scores_key: str = field( # not used in UltraFeedback
        default="scores",
        metadata={"help": "Key used to read per-dimension scores from each completion item."},
    )
    listwise_annotations_key: str = field(
        default="annotations",
        metadata={"help": "Fallback key used to read per-dimension scores from each completion item."},
    )
    listwise_min_responses: int = field(
        default=2,
        metadata={"help": "Drop prompts with fewer than this number of scored responses."},
    )
    listwise_seed: int = field(
        default=0,
        metadata={"help": "Random seed for listwise subsampling operations."},
    )
    listwise_use_cyclic_filter: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable rotated-cycle filtering for listwise conversion. "
                "When enabled, listwise rows are built only from cyclic-matching candidates."
            )
        },
    )
    pairwise_from_listwise_strategy: str = field(
        default="extreme",
        metadata={
            "help": (
                "When dataset_format='pairwise' and the loaded dataset has listwise ranking columns, "
                "convert each ranking row to pairwise DPO rows. Choices: 'extreme' uses top vs bottom; "
                "'all_pairs' emits every ordered pair from the observed ranking."
            )
        },
    )
    dataset_downsample_ratio: float = field(
        default=1.0,
        metadata={
            "help": (
                "Deterministically downsample loaded dataset splits before listwise/pairwise conversion. "
                "Use values in (0, 1]; default 1 keeps all rows."
            )
        },
    )
    dataset_downsample_seed: int = field(
        default=0,
        metadata={"help": "Seed for deterministic dataset downsampling."},
    )
    dataset_downsample_splits: Optional[list[str]] = field(
        default_factory=lambda: ["train", "validation", "test"],
        metadata={
            "help": (
                "Splits to downsample when dataset_downsample_ratio < 1. "
                "Use ['all'] to downsample every split."
            )
        },
    )
    dataset_downsample_group_key: str = field(
        default="source_index",
        metadata={
            "help": (
                "Column used for grouped downsampling when present. "
                "Default source_index keeps all rows derived from one prompt together."
            )
        },
    )
    run_ranking_eval: bool = field(
        default=True,
        metadata={
            "help": (
                "If True and ranking/listwise splits are available, run offline ranking evaluation on "
                "train/validation/test after training."
            )
        },
    )
    ranking_eval_during_training: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, run ranking/listwise evaluation on the configured eval split each time "
                "Trainer.evaluate() runs, so ranking accuracy is visible during training."
            )
        },
    )
    eval_only_ranking: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, skip training and run ranking evaluation only using a loaded checkpoint/model."
            )
        },
    )
    eval_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional checkpoint path to load before eval_only_ranking. "
                "Accepts Trainer checkpoint-* directory or final saved model directory."
            )
        },
    )
    eval_ranking_split: str = field(
        default="test",
        metadata={"help": "Ranking split to evaluate in eval_only_ranking mode. Choices: train/validation/test."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_format not in {"pairwise", "listwise"}:
            raise ValueError("`dataset_format` must be either 'pairwise' or 'listwise'")
        if self.pairwise_from_listwise_strategy not in {"extreme", "all_pairs"}:
            raise ValueError("`pairwise_from_listwise_strategy` must be either 'extreme' or 'all_pairs'")
        if self.listwise_num_responses < 2:
            raise ValueError("`listwise_num_responses` must be >= 2")
        if self.listwise_num_generated_responses < 0:
            raise ValueError("`listwise_num_generated_responses` must be >= 0")
        if self.listwise_min_responses < 2:
            raise ValueError("`listwise_min_responses` must be >= 2")
        if self.dataset_downsample_ratio <= 0 or self.dataset_downsample_ratio > 1:
            raise ValueError("`dataset_downsample_ratio` must be in (0, 1].")
        if isinstance(self.dataset_downsample_splits, str):
            self.dataset_downsample_splits = [
                split.strip()
                for split in self.dataset_downsample_splits.split(",")
                if split.strip()
            ]
        if self.dataset_downsample_splits is not None:
            if not isinstance(self.dataset_downsample_splits, list):
                raise ValueError("`dataset_downsample_splits` must be a list when provided")
            if any(not isinstance(split, str) or not split for split in self.dataset_downsample_splits):
                raise ValueError("`dataset_downsample_splits` entries must be non-empty strings")
        if self.preference_dimensions is not None:
            if not isinstance(self.preference_dimensions, list):
                raise ValueError("`preference_dimensions` must be a list when provided")
            if len(self.preference_dimensions) < 1:
                raise ValueError("`preference_dimensions` must contain at least 1 dimension")
            if len(set(self.preference_dimensions)) != len(self.preference_dimensions):
                raise ValueError("`preference_dimensions` must not contain duplicates")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    listwise: bool = field(default=False, metadata={"help": "Enable listwise DPO training."})
    listwise_beta: Optional[float] = field(
        default=None,
        metadata={"help": "Optional beta override for listwise PL loss. Falls back to DPO beta when unset."},
    )


@dataclass
class MixturePLConfig(DPOConfig):
    """
    Configuration for Mixture of Plackett-Luce (MoPL) models in DPO training.

    When enabled, learns to cluster rankings by preference dimension with a mixture PL
    objective while jointly optimizing the DPO objective.
    """

    use_mixture: bool = field(
        default=False,
        metadata={"help": "Enable mixture of Plackett-Luce (MoPL) clustering during DPO training."},
    )
    mixture_objective: str = field(
        default="pl",
        metadata={"help": "Mixture objective to use when use_mixture=True. Choices: 'pl' or 'bt'."},
    )
    num_clusters: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of ranking clusters. If None, auto-detected from preference_dimensions. "
                "Must be provided when preference_dimensions is not set."
            )
        },
    )
    em_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for E-step soft assignments (lower = harder assignments)."},
    )
    mixture_nll_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for mixture PL loss in hybrid objective: DPO_loss + weight * mixture_em_nll."},
    )
    mixture_training_mode: str = field(
        default="hybrid_dpo_em",
        metadata={
            "help": (
                "Mixture training mode: 'hybrid_dpo_em' uses DPO_loss + weight * mixture_em_nll; "
                "'em_only' optimizes only the EM expected complete-data NLL."
            )
        },
    )
    m_step_updates: int = field(
        default=1,
        metadata={"help": "Number of optimizer updates per E-step when mixture_training_mode='em_only'."},
    )
    mixture_reward_backend: str = field(
        default="head",
        metadata={
            "help": (
                "Reward backend for mixture components: 'head' uses explicit scalar reward heads; "
                "'lora' creates one LoRA adapter per cluster and uses adapter DPO utilities as rewards."
            )
        },
    )
    mixture_lora_adapter_prefix: str = field(
        default="mixture_cluster",
        metadata={"help": "Prefix for per-cluster LoRA adapter names when mixture_reward_backend='lora'."},
    )
    use_contextual_router: bool = field(
        default=False,
        metadata={"help": "If True, use contextual router (MLP on hidden state). If False, use global mixture weights."},
    )
    router_hidden_size: int = field(
        default=256,
        metadata={"help": "Hidden dimension of router MLP (only used if use_contextual_router=True)."},
    )
    use_auxiliary_dimension_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, add auxiliary loss to align router assignments with ground-truth preference_dimension. "
                "Only used during training (not evaluation). Scales by mixture_nll_weight."
            )
        },
    )
    use_closed_form_router_prior_update: bool = field(
        default=False,
        metadata={
            "help": (
                "If True and use_contextual_router=False, update the global router prior in closed form "
                "from EM responsibilities gamma using a batchwise log-mean prior."
            )
        },
    )
    log_cluster_metrics: bool = field(
        default=True,
        metadata={"help": "If True, compute and log cluster assignment accuracy during validation."},
    )
    use_linear_reward_approx: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use anchor-based first-order reward approximation for listwise/mixture training. "
                "Standard pairwise DPO remains exact."
            )
        },
    )
    linear_approx_num_anchors: int = field(
        default=2,
        metadata={"help": "Number of valid candidate responses to score exactly per row when approximation is enabled."},
    )
    linear_approx_gradient_mode: str = field(
        default="stop_gradient",
        metadata={"help": "Gradient mode for non-anchor estimates. Currently only 'stop_gradient' is supported."},
    )
    linear_approx_exact_eval: bool = field(
        default=True,
        metadata={"help": "If True, validation/evaluation uses exact scoring even when approximation is enabled."},
    )
    linear_approx_ref_mode: str = field(
        default="input_gradient",
        metadata={
            "help": (
                "Reference handling for linear approximation. 'input_gradient' approximates the full "
                "DPO utility gradient, while 'exact_score' subtracts exact no-grad reference scores "
                "and ignores reference input gradients."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.use_mixture:
            if self.num_clusters is None:
                # Try to infer from preference_dimensions if available
                # This will be validated later when we have access to ScriptArguments
                pass
            elif self.num_clusters < 1:
                raise ValueError("`num_clusters` must be >= 1 when use_mixture=True")
            if self.mixture_objective not in {"pl", "bt"}:
                raise ValueError("`mixture_objective` must be either 'pl' or 'bt'")

            if self.em_temperature <= 0:
                raise ValueError("`em_temperature` must be positive")
            if self.mixture_nll_weight < 0:
                raise ValueError("`mixture_nll_weight` must be >= 0")
            if self.mixture_training_mode not in {"hybrid_dpo_em", "em_only"}:
                raise ValueError("`mixture_training_mode` must be either 'hybrid_dpo_em' or 'em_only'")
            if self.m_step_updates < 1:
                raise ValueError("`m_step_updates` must be >= 1")
            if self.mixture_reward_backend not in {"head", "lora"}:
                raise ValueError("`mixture_reward_backend` must be either 'head' or 'lora'")
            if not self.mixture_lora_adapter_prefix:
                raise ValueError("`mixture_lora_adapter_prefix` must be non-empty")
            if self.router_hidden_size < 1:
                raise ValueError("`router_hidden_size` must be >= 1")

        if self.use_linear_reward_approx:
            if self.linear_approx_num_anchors < 1:
                raise ValueError("`linear_approx_num_anchors` must be >= 1")
            if self.linear_approx_gradient_mode != "stop_gradient":
                raise ValueError("Only `linear_approx_gradient_mode='stop_gradient'` is currently supported")
            if self.linear_approx_ref_mode not in {"input_gradient", "exact_score"}:
                raise ValueError("`linear_approx_ref_mode` must be either 'input_gradient' or 'exact_score'")


if hasattr(trl, "ORPOConfig"):

    @dataclass
    class ORPOConfig(trl.ORPOConfig):
        """
        args for callbacks, benchmarks etc
        """

        chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
else:

    @dataclass
    class ORPOConfig:
        """
        Placeholder for environments where TRL no longer exposes ORPOConfig.
        Allows importing non-ORPO workflows while providing a clear runtime error
        if ORPO is selected.
        """

        chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})

        def __post_init__(self):
            raise RuntimeError(
                "`trl.ORPOConfig` is not available in the installed TRL version. "
                "Install a compatible TRL release (e.g. `trl<1.0`) to run ORPO."
            )
