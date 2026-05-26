#!/usr/bin/env python
"""Evaluate generation win-rate against a base model using DPO-utility reward adapters."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument(
        "--sample_strategy",
        choices=("balanced_by_dimension", "first_n", "random"),
        default="balanced_by_dimension",
    )
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--policy_adapter_path", required=True)
    parser.add_argument(
        "--policy_adapter_mode",
        choices=("single", "dimension_mapped_mixture"),
        default="single",
    )
    parser.add_argument(
        "--reward_adapter_map_json",
        default=None,
        help="Path to a JSON object, or an inline JSON object, mapping dimensions to reward LoRA adapter paths.",
    )
    parser.add_argument(
        "--reward_adapter",
        action="append",
        default=[],
        metavar="DIMENSION=PATH",
        help="Reward LoRA adapter mapping. May be repeated.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--reward_beta", type=float, default=0.01)
    parser.add_argument("--tie_epsilon", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--prompt_format", choices=("raw", "chat_auto"), default="raw")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args(argv)


def _torch_dtype_from_arg(value: str) -> Any:
    if value == "auto":
        return "auto"
    if not hasattr(torch, value):
        raise ValueError(f"Unknown torch dtype {value!r}. Use 'auto' or a torch dtype name like 'bfloat16'.")
    return getattr(torch, value)


def _safe_adapter_name(prefix: str, value: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_") or "unknown"
    return f"{prefix}_{safe}"


def _model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _adapter_context(model: torch.nn.Module, adapter_name: str | None):
    if adapter_name is None:
        if hasattr(model, "disable_adapter"):
            return model.disable_adapter()
        return nullcontext()
    if not hasattr(model, "set_adapter"):
        raise ValueError("Requested an adapter but model does not support set_adapter().")
    model.set_adapter(adapter_name)
    return nullcontext()


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype_from_arg(args.torch_dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return kwargs


def load_dataset_dict(dataset_name: str, dataset_config: str | None = None) -> DatasetDict:
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, dataset_config)
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    if isinstance(dataset, DatasetDict):
        return dataset
    raise TypeError(f"Unsupported dataset object loaded from {dataset_name!r}: {type(dataset)}")


def select_eval_examples(
    dataset: Dataset,
    *,
    max_examples: int | None,
    sample_strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in dataset]
    if max_examples is None or max_examples >= len(rows):
        return rows
    if max_examples < 0:
        raise ValueError("`max_examples` must be non-negative when provided.")
    if sample_strategy == "first_n":
        return rows[:max_examples]
    if sample_strategy == "random":
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        return [rows[idx] for idx in indices[:max_examples]]
    if sample_strategy != "balanced_by_dimension":
        raise ValueError(f"Unknown sample strategy: {sample_strategy}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dimension_order: list[str] = []
    for row in rows:
        dimension = str(row.get("preference_dimension", "unknown"))
        if dimension not in groups:
            dimension_order.append(dimension)
        groups[dimension].append(row)

    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < max_examples:
        made_progress = False
        for dimension in dimension_order:
            bucket = groups[dimension]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                made_progress = True
                if len(selected) == max_examples:
                    break
        if not made_progress:
            break
        cursor += 1
    return selected


def parse_reward_adapter_map(
    *,
    reward_adapter_map_json: str | None,
    reward_adapters: list[str] | None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if reward_adapter_map_json:
        source = Path(reward_adapter_map_json)
        if source.exists():
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            payload = json.loads(reward_adapter_map_json)
        if not isinstance(payload, dict):
            raise ValueError("`reward_adapter_map_json` must contain a JSON object.")
        mapping.update({str(key): str(value) for key, value in payload.items()})

    for item in reward_adapters or []:
        if "=" not in item:
            raise ValueError(f"Invalid --reward_adapter {item!r}; expected DIMENSION=PATH.")
        dimension, adapter_path = item.split("=", 1)
        dimension = dimension.strip()
        adapter_path = adapter_path.strip()
        if not dimension or not adapter_path:
            raise ValueError(f"Invalid --reward_adapter {item!r}; dimension and path must be non-empty.")
        if dimension in mapping:
            raise ValueError(f"Duplicate reward adapter mapping for dimension {dimension!r}.")
        mapping[dimension] = adapter_path
    return mapping


def validate_reward_adapter_map(required_dimensions: set[str], mapping: dict[str, str]) -> None:
    missing = sorted(required_dimensions.difference(mapping))
    if missing:
        raise ValueError(f"Missing reward adapter path(s) for dimension(s): {missing}")


def _persona_dimension_index(dimension: str) -> int:
    match = re.fullmatch(r"persona_(\d+)", str(dimension))
    if not match:
        raise ValueError(
            f"Cannot infer mixture adapter index from dimension {dimension!r}. "
            "Expected names like 'persona_0003'."
        )
    return int(match.group(1))


def resolve_policy_adapter_path(policy_adapter_path: str, policy_adapter_mode: str, dimension: str) -> str:
    if policy_adapter_mode == "single":
        return policy_adapter_path
    if policy_adapter_mode != "dimension_mapped_mixture":
        raise ValueError(f"Unknown policy adapter mode: {policy_adapter_mode}")
    return str(Path(policy_adapter_path) / f"mixture_cluster_{_persona_dimension_index(dimension)}")


def build_sequence(
    tokenizer,
    prompt: str,
    response: str,
    *,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids = response_ids + [eos_id]

    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        original_prompt_len = len(prompt_ids)
        if overflow < original_prompt_len:
            prompt_ids = prompt_ids[overflow:]
        else:
            prompt_ids = []
            response_ids = response_ids[overflow - original_prompt_len:]
        input_ids = prompt_ids + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def pad_sequences(tokenizer, sequences: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
    max_len = max(len(item["input_ids"]) for item in sequences)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in sequences:
        pad_len = max_len - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
        batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
        batch["labels"].append(item["labels"] + [-100] * pad_len)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logps = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    return (token_logps * valid).sum(dim=-1)


class PolicyGenerator:
    def __init__(self, args: argparse.Namespace, dimensions: set[str]):
        from peft import PeftModel

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, **_model_kwargs(args))
        self.dimension_to_adapter: dict[str, str] = {}
        self.dimension_to_path: dict[str, str] = {}

        ordered_dimensions = sorted(dimensions)
        first_dimension = ordered_dimensions[0]
        first_path = resolve_policy_adapter_path(args.policy_adapter_path, args.policy_adapter_mode, first_dimension)
        first_adapter = _safe_adapter_name("policy", first_dimension if args.policy_adapter_mode != "single" else "single")
        self.model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first_adapter)

        if args.policy_adapter_mode == "single":
            for dimension in ordered_dimensions:
                self.dimension_to_adapter[dimension] = first_adapter
                self.dimension_to_path[dimension] = first_path
        else:
            self.dimension_to_adapter[first_dimension] = first_adapter
            self.dimension_to_path[first_dimension] = first_path
            for dimension in ordered_dimensions[1:]:
                adapter_path = resolve_policy_adapter_path(args.policy_adapter_path, args.policy_adapter_mode, dimension)
                adapter_name = _safe_adapter_name("policy", dimension)
                self.model.load_adapter(adapter_path, adapter_name=adapter_name)
                self.dimension_to_adapter[dimension] = adapter_name
                self.dimension_to_path[dimension] = adapter_path

        self.model.eval()

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        if self.args.prompt_format == "raw":
            return prompts
        if getattr(self.tokenizer, "chat_template", None) is None:
            LOGGER.warning("prompt_format=chat_auto requested, but tokenizer has no chat_template; using raw prompts.")
            return prompts
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    @torch.inference_mode()
    def generate(self, prompts: list[str], *, adapter_name: str | None) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), self.args.batch_size):
            batch_prompts = prompts[start : start + self.args.batch_size]
            formatted = self._format_prompts(batch_prompts)
            inputs = self.tokenizer(formatted, return_tensors="pt", padding=True)
            device = _model_input_device(self.model)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with _adapter_context(self.model, adapter_name):
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=self.args.temperature > 0,
                    temperature=self.args.temperature if self.args.temperature > 0 else None,
                    top_p=self.args.top_p,
                    max_new_tokens=self.args.max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_width = inputs["input_ids"].shape[1]
            for row_idx in range(len(batch_prompts)):
                generated_ids = output_ids[row_idx, input_width:]
                outputs.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        return outputs

    def generate_base(self, prompts: list[str]) -> list[str]:
        return self.generate(prompts, adapter_name=None)

    def generate_policy_by_dimension(self, examples: list[dict[str, Any]]) -> list[str]:
        outputs: list[str | None] = [None] * len(examples)
        indices_by_adapter: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(examples):
            dimension = str(row.get("preference_dimension", "unknown"))
            indices_by_adapter[self.dimension_to_adapter[dimension]].append(idx)

        for adapter_name, indices in indices_by_adapter.items():
            prompts = [str(examples[idx].get("prompt", "")) for idx in indices]
            generated = self.generate(prompts, adapter_name=adapter_name)
            for idx, response in zip(indices, generated):
                outputs[idx] = response
        return [str(response) for response in outputs]


class DPOUtilityRewardScorer:
    def __init__(self, args: argparse.Namespace, reward_adapter_map: dict[str, str]):
        from peft import PeftModel

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, **_model_kwargs(args))
        ordered_items = sorted(reward_adapter_map.items())
        first_dimension, first_path = ordered_items[0]
        first_adapter = _safe_adapter_name("reward", first_dimension)
        self.model = PeftModel.from_pretrained(base_model, first_path, adapter_name=first_adapter)
        self.dimension_to_adapter = {first_dimension: first_adapter}
        self.dimension_to_path = {first_dimension: first_path}
        for dimension, adapter_path in ordered_items[1:]:
            adapter_name = _safe_adapter_name("reward", dimension)
            self.model.load_adapter(adapter_path, adapter_name=adapter_name)
            self.dimension_to_adapter[dimension] = adapter_name
            self.dimension_to_path[dimension] = adapter_path
        self.model.eval()

    @torch.inference_mode()
    def score_dimension(self, dimension: str, pairs: list[tuple[str, str]]) -> list[float]:
        adapter_name = self.dimension_to_adapter[dimension]
        rewards: list[float] = []
        for start in range(0, len(pairs), self.args.batch_size):
            batch_pairs = pairs[start : start + self.args.batch_size]
            sequences = [
                build_sequence(
                    self.tokenizer,
                    prompt,
                    response,
                    max_length=self.args.max_length,
                    max_prompt_length=self.args.max_prompt_length,
                )
                for prompt, response in batch_pairs
            ]
            batch = pad_sequences(self.tokenizer, sequences)
            device = _model_input_device(self.model)
            batch = {key: value.to(device) for key, value in batch.items()}

            with _adapter_context(self.model, adapter_name):
                policy_logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
            with _adapter_context(self.model, None):
                base_logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
            policy_logps = sequence_logps(policy_logits, batch["labels"])
            base_logps = sequence_logps(base_logits, batch["labels"])
            batch_rewards = self.args.reward_beta * (policy_logps - base_logps)
            rewards.extend(float(value) for value in batch_rewards.detach().float().cpu().tolist())
        return rewards

    def score_records(self, records: list[dict[str, Any]]) -> None:
        items_by_dimension: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for idx, record in enumerate(records):
            dimension = str(record["preference_dimension"])
            prompt = str(record["prompt"])
            items_by_dimension[dimension].append((idx, prompt, str(record["base_response"])))
            items_by_dimension[dimension].append((idx, prompt, str(record["policy_response"])))

        for dimension, items in tqdm(
            sorted(items_by_dimension.items()),
            desc="Scoring responses",
            unit="dimension",
            disable=self.args.disable_tqdm,
        ):
            pairs = [(prompt, response) for _, prompt, response in items]
            scores = self.score_dimension(dimension, pairs)
            for offset in range(0, len(items), 2):
                record_idx = items[offset][0]
                records[record_idx]["base_reward_score"] = scores[offset]
                records[record_idx]["policy_reward_score"] = scores[offset + 1]


def aggregate_metrics(records: list[dict[str, Any]], *, tie_epsilon: float = 0.0) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
        wins = losses = ties = 0
        margins = []
        for row in rows:
            margin = float(row["policy_reward_score"]) - float(row["base_reward_score"])
            margins.append(margin)
            if margin > tie_epsilon:
                wins += 1
                row["winner"] = "policy"
            elif margin < -tie_epsilon:
                losses += 1
                row["winner"] = "base"
            else:
                ties += 1
                row["winner"] = "tie"
        total = len(rows)
        return {
            "num_examples": float(total),
            "win_rate": wins / max(total, 1),
            "loss_rate": losses / max(total, 1),
            "tie_rate": ties / max(total, 1),
            "mean_score_margin": sum(margins) / max(total, 1),
        }

    metrics: dict[str, Any] = summarize(records)
    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dimension[str(record["preference_dimension"])].append(record)
    metrics["by_dimension"] = {
        dimension: summarize(rows)
        for dimension, rows in sorted(by_dimension.items())
    }
    return metrics


def prepare_output_dir(output_dir: str, overwrite: bool) -> Path:
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists and is not a directory. Pass --overwrite to replace it.")
        output_path.unlink()
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists and is non-empty. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    omitted = {"prompt", "responses", "scores"}
    return {key: value for key, value in row.items() if key not in omitted}


def write_outputs(records: list[dict[str, Any]], metrics: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")
    if args.max_new_tokens < 1:
        raise ValueError("`max_new_tokens` must be at least 1.")
    if args.temperature < 0:
        raise ValueError("`temperature` must be non-negative.")
    if args.tie_epsilon < 0:
        raise ValueError("`tie_epsilon` must be non-negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    reward_adapter_map = parse_reward_adapter_map(
        reward_adapter_map_json=args.reward_adapter_map_json,
        reward_adapters=args.reward_adapter,
    )
    if not reward_adapter_map:
        raise ValueError("Provide at least one reward adapter via --reward_adapter_map_json or --reward_adapter.")

    LOGGER.info("Loading dataset %s", args.dataset_name)
    dataset_dict = load_dataset_dict(args.dataset_name, args.dataset_config)
    if args.split not in dataset_dict:
        raise ValueError(f"Split {args.split!r} not found. Available splits: {list(dataset_dict.keys())}")
    examples = select_eval_examples(
        dataset_dict[args.split],
        max_examples=args.max_examples,
        sample_strategy=args.sample_strategy,
        seed=args.seed,
    )
    if not examples:
        raise ValueError("No examples selected for evaluation.")
    dimensions = {str(row.get("preference_dimension", "unknown")) for row in examples}
    validate_reward_adapter_map(dimensions, reward_adapter_map)

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)

    LOGGER.info("Loading policy model and adapter(s).")
    generator = PolicyGenerator(args, dimensions)
    prompts = [str(row.get("prompt", "")) for row in examples]
    base_responses = generator.generate_base(prompts)
    policy_responses = generator.generate_policy_by_dimension(examples)

    records = []
    generation_settings = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "prompt_format": args.prompt_format,
    }
    for row, base_response, policy_response in zip(examples, base_responses, policy_responses):
        dimension = str(row.get("preference_dimension", "unknown"))
        records.append(
            {
                "prompt": str(row.get("prompt", "")),
                "preference_dimension": dimension,
                "base_response": base_response,
                "policy_response": policy_response,
                "policy_adapter_path": generator.dimension_to_path[dimension],
                "reward_adapter_path": reward_adapter_map[dimension],
                "generation_settings": generation_settings,
                "metadata": _record_metadata(row),
            }
        )

    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info("Loading reward model and adapter(s).")
    scorer = DPOUtilityRewardScorer(args, reward_adapter_map)
    scorer.score_records(records)
    metrics = aggregate_metrics(records, tie_epsilon=args.tie_epsilon)
    metrics.update(
        {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "policy_adapter_path": args.policy_adapter_path,
            "policy_adapter_mode": args.policy_adapter_mode,
            "reward_beta": args.reward_beta,
            "tie_epsilon": args.tie_epsilon,
        }
    )
    write_outputs(records, metrics, output_dir)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    args = parse_args()
    metrics = evaluate(args)
    print(
        "Generation win-rate: "
        f"n={int(metrics['num_examples'])} "
        f"win={metrics['win_rate']:.4f} "
        f"loss={metrics['loss_rate']:.4f} "
        f"tie={metrics['tie_rate']:.4f} "
        f"mean_margin={metrics['mean_score_margin']:.4f}"
    )


if __name__ == "__main__":
    main()
