"""Toy end-to-end smoke test for MixtureDPOTrainer.

Runs a minimal training loop on a synthetic listwise dataset using a tiny GPT-2 model.
"""

import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import GPT2Config, GPT2LMHeadModel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from alignment import MixtureDPOTrainer, MixtureEMDPOTrainer, MixturePLConfig


class TinyTokenizer:
    """Minimal tokenizer for synthetic tests (character-level, deterministic)."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return {"input_ids": ids}

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)


def build_toy_listwise_dataset() -> Dataset:
    examples = [
        {
            "prompt": "Explain why regular exercise helps health.",
            "responses": [
                "Exercise improves cardiovascular function and mood.",
                "Exercise can be useful in many contexts.",
                "I do not know.",
            ],
            "preference_dimension": "helpfulness",
        },
        {
            "prompt": "How should someone safely store passwords?",
            "responses": [
                "Use a password manager and unique passwords per site.",
                "Try to remember one long password for all accounts.",
                "Write them on paper next to your laptop.",
            ],
            "preference_dimension": "helpfulness",
        },
        {
            "prompt": "Should I mix bleach and vinegar for cleaning?",
            "responses": [
                "No. Mixing them can release toxic chlorine gas.",
                "Maybe, if the room has airflow.",
                "Yes, it always gives stronger cleaning power.",
            ],
            "preference_dimension": "harmlessness",
        },
        {
            "prompt": "Is it safe to share my bank PIN with a friend?",
            "responses": [
                "No, never share your PIN with anyone.",
                "Only share if you trust them a lot.",
                "Yes, that is fine for emergencies.",
            ],
            "preference_dimension": "harmlessness",
        },
    ]
    return Dataset.from_list(examples)


def run_toy_training_smoke_test() -> None:
    torch.manual_seed(0)

    tokenizer = TinyTokenizer(vocab_size=128)

    model_config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(model_config)
    ref_model = GPT2LMHeadModel(model_config)
    ref_model.load_state_dict(model.state_dict())

    dataset = build_toy_listwise_dataset()

    training_args = MixturePLConfig(
        output_dir="/tmp/mixture_dpo_toy_smoke",
        bf16=False,
        remove_unused_columns=False,
        report_to=[],
        max_steps=2,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        max_length=96,
        max_prompt_length=48,
        use_mixture=True,
        num_clusters=2,
        mixture_nll_weight=0.1,
        use_contextual_router=True,
        router_hidden_size=32,
        em_temperature=1.0,
        log_cluster_metrics=True,
    )

    trainer = MixtureDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        mixture_config=training_args,
        listwise_beta=0.1,
    )
    mixture_params = [
        param for name, param in trainer.model.named_parameters() if name.startswith("mixture_pl_head.")
    ]
    if not mixture_params:
        raise RuntimeError("Mixture PL head parameters are not attached to the trained model.")

    result = trainer.train()
    optimizer_param_ids = {id(param) for group in trainer.optimizer.param_groups for param in group["params"]}
    if any(id(param) not in optimizer_param_ids for param in mixture_params if param.requires_grad):
        raise RuntimeError("Trainable mixture PL head parameters are missing from the optimizer.")

    trainer.save_model(training_args.output_dir)
    mixture_head_path = os.path.join(training_args.output_dir, "mixture_pl_head.pt")
    if not os.path.exists(mixture_head_path):
        raise RuntimeError(f"Mixture PL head checkpoint was not saved: {mixture_head_path}")

    if not torch.isfinite(torch.tensor(result.training_loss)):
        raise RuntimeError(f"Training loss is not finite: {result.training_loss}")

    print("=" * 80)
    print("TOY MIXTURE DPO SMOKE TEST")
    print("=" * 80)
    print(f"Train runtime: {result.metrics.get('train_runtime', 'n/a')}")
    print(f"Train loss: {result.training_loss:.6f}")
    print(f"Train steps: {result.metrics.get('train_steps_per_second', 'n/a')}")
    print("✓ MixtureDPOTrainer toy training loop completed successfully")


def run_toy_lora_adapter_training_smoke_test() -> None:
    torch.manual_seed(0)

    tokenizer = TinyTokenizer(vocab_size=128)
    model_config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(model_config)
    dataset = build_toy_listwise_dataset()

    training_args = MixturePLConfig(
        output_dir="/tmp/mixture_dpo_toy_lora_smoke",
        bf16=False,
        remove_unused_columns=False,
        report_to=[],
        max_steps=1,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        max_length=96,
        max_prompt_length=48,
        use_mixture=True,
        num_clusters=2,
        mixture_nll_weight=0.1,
        mixture_reward_backend="lora",
        use_contextual_router=False,
        em_temperature=1.0,
        log_cluster_metrics=True,
    )
    peft_config = LoraConfig(
        r=2,
        lora_alpha=2,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )

    trainer = MixtureDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        mixture_config=training_args,
        listwise_beta=0.1,
    )
    if len(trainer.mixture_adapter_names) != training_args.num_clusters:
        raise RuntimeError("Incorrect number of mixture LoRA adapters was created.")
    for adapter_name in trainer.mixture_adapter_names:
        if adapter_name not in trainer.model.peft_config:
            raise RuntimeError(f"Missing mixture LoRA adapter: {adapter_name}")

    result = trainer.train()
    optimizer_param_names = {
        name
        for group in trainer.optimizer.param_groups
        for param in group["params"]
        for name, named_param in trainer.model.named_parameters()
        if param is named_param
    }
    for adapter_name in trainer.mixture_adapter_names:
        if not any(f".{adapter_name}." in name for name in optimizer_param_names):
            raise RuntimeError(f"Mixture LoRA adapter is missing from the optimizer: {adapter_name}")

    trainer.save_model(training_args.output_dir)
    mixture_router_path = os.path.join(training_args.output_dir, "mixture_router.pt")
    if not os.path.exists(mixture_router_path):
        raise RuntimeError(f"Mixture router checkpoint was not saved: {mixture_router_path}")
    if not torch.isfinite(torch.tensor(result.training_loss)):
        raise RuntimeError(f"LoRA mixture training loss is not finite: {result.training_loss}")

    print("=" * 80)
    print("TOY MIXTURE DPO LORA SMOKE TEST")
    print("=" * 80)
    print(f"Train loss: {result.training_loss:.6f}")
    print("✓ MixtureDPOTrainer LoRA-adapter backend completed successfully")


def run_toy_em_only_training_smoke_test() -> None:
    torch.manual_seed(0)

    tokenizer = TinyTokenizer(vocab_size=128)
    model_config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(model_config)
    ref_model = GPT2LMHeadModel(model_config)
    ref_model.load_state_dict(model.state_dict())

    training_args = MixturePLConfig(
        output_dir="/tmp/mixture_dpo_toy_em_only_smoke",
        bf16=False,
        remove_unused_columns=False,
        report_to=[],
        max_steps=2,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        max_length=96,
        max_prompt_length=48,
        use_mixture=True,
        num_clusters=2,
        mixture_training_mode="em_only",
        m_step_updates=2,
        mixture_reward_backend="head",
        mixture_nll_weight=0.1,
        use_contextual_router=True,
        router_hidden_size=32,
        em_temperature=1.0,
        log_cluster_metrics=True,
    )

    trainer = MixtureEMDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=build_toy_listwise_dataset(),
        eval_dataset=build_toy_listwise_dataset(),
        processing_class=tokenizer,
        mixture_config=training_args,
        listwise_beta=0.1,
    )

    result = trainer.train()
    if result.global_step != training_args.max_steps:
        raise RuntimeError(f"Expected {training_args.max_steps} EM optimizer steps, got {result.global_step}.")
    if not torch.isfinite(torch.tensor(result.training_loss)):
        raise RuntimeError(f"EM-only training loss is not finite: {result.training_loss}")

    logged_keys = set()
    for log_entry in trainer.state.log_history:
        logged_keys.update(log_entry)
    required_log_keys = {
        "mixture/cluster_acc",
        "mixture/cluster_acc_raw",
        "listwise/top1_acc",
        "listwise/pairwise_acc",
        "listwise/utility_first",
        "listwise/utility_last",
        "listwise/utility_mean",
    }
    missing_log_keys = required_log_keys - logged_keys
    if missing_log_keys:
        raise RuntimeError(f"Missing EM-only trainer log metrics: {sorted(missing_log_keys)}")

    print("=" * 80)
    print("TOY MIXTURE DPO EM-ONLY SMOKE TEST")
    print("=" * 80)
    print(f"Train loss: {result.training_loss:.6f}")
    print("✓ MixtureEMDPOTrainer EM-only loop completed successfully")


def run_toy_lora_em_only_training_smoke_test() -> None:
    torch.manual_seed(0)

    tokenizer = TinyTokenizer(vocab_size=128)
    model_config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(model_config)

    training_args = MixturePLConfig(
        output_dir="/tmp/mixture_dpo_toy_lora_em_only_smoke",
        bf16=False,
        remove_unused_columns=False,
        report_to=[],
        max_steps=2,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        max_length=96,
        max_prompt_length=48,
        use_mixture=True,
        num_clusters=2,
        mixture_training_mode="em_only",
        m_step_updates=2,
        mixture_reward_backend="lora",
        mixture_nll_weight=0.1,
        use_contextual_router=False,
        router_hidden_size=32,
        em_temperature=1.0,
        log_cluster_metrics=True,
    )
    peft_config = LoraConfig(
        r=2,
        lora_alpha=2,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )

    trainer = MixtureEMDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=build_toy_listwise_dataset(),
        eval_dataset=build_toy_listwise_dataset(),
        processing_class=tokenizer,
        peft_config=peft_config,
        mixture_config=training_args,
        listwise_beta=0.1,
    )

    result = trainer.train()
    if result.global_step != training_args.max_steps:
        raise RuntimeError(f"Expected {training_args.max_steps} LoRA EM optimizer steps, got {result.global_step}.")
    if not torch.isfinite(torch.tensor(result.training_loss)):
        raise RuntimeError(f"LoRA EM-only training loss is not finite: {result.training_loss}")

    logged_keys = set()
    for log_entry in trainer.state.log_history:
        logged_keys.update(log_entry)
    required_log_keys = {
        "mixture/router_em_nll",
        "mixture/component_em_nll",
        "listwise/top1_acc",
        "listwise/pairwise_acc",
    }
    missing_log_keys = required_log_keys - logged_keys
    if missing_log_keys:
        raise RuntimeError(f"Missing LoRA EM-only trainer log metrics: {sorted(missing_log_keys)}")

    print("=" * 80)
    print("TOY MIXTURE DPO LORA EM-ONLY SMOKE TEST")
    print("=" * 80)
    print(f"Train loss: {result.training_loss:.6f}")
    print("✓ MixtureEMDPOTrainer LoRA EM-only loop completed successfully")


def run_toy_lora_em_only_linear_approx_smoke_test() -> None:
    for ref_mode in ("input_gradient", "exact_score"):
        torch.manual_seed(0)

        tokenizer = TinyTokenizer(vocab_size=128)
        model_config = GPT2Config(
            vocab_size=128,
            n_positions=128,
            n_ctx=128,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
        model = GPT2LMHeadModel(model_config)

        training_args = MixturePLConfig(
            output_dir=f"/tmp/mixture_dpo_toy_lora_em_only_linear_approx_{ref_mode}",
            bf16=False,
            remove_unused_columns=False,
            report_to=[],
            max_steps=1,
            save_strategy="no",
            eval_strategy="no",
            logging_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            max_length=96,
            max_prompt_length=48,
            use_mixture=True,
            num_clusters=2,
            mixture_training_mode="em_only",
            m_step_updates=1,
            mixture_reward_backend="lora",
            mixture_nll_weight=0.1,
            use_contextual_router=False,
            router_hidden_size=32,
            em_temperature=1.0,
            log_cluster_metrics=True,
            use_linear_reward_approx=True,
            linear_approx_num_anchors=1,
            linear_approx_ref_mode=ref_mode,
        )
        peft_config = LoraConfig(
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        )

        trainer = MixtureEMDPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=build_toy_listwise_dataset(),
            eval_dataset=build_toy_listwise_dataset(),
            processing_class=tokenizer,
            peft_config=peft_config,
            mixture_config=training_args,
            listwise_beta=0.1,
        )

        active_adapter = {"name": None}
        original_set_active_adapter = trainer._set_active_adapter

        def tracked_set_active_adapter(adapter_name: str) -> None:
            active_adapter["name"] = adapter_name
            original_set_active_adapter(adapter_name)

        original_forward = trainer.model.forward
        mixture_forward_batches = []

        def tracked_forward(*args, **kwargs):
            if active_adapter["name"] in set(trainer.mixture_adapter_names):
                if kwargs.get("inputs_embeds") is not None:
                    mixture_forward_batches.append(("inputs_embeds", int(kwargs["inputs_embeds"].shape[0])))
                elif kwargs.get("input_ids") is not None:
                    mixture_forward_batches.append(("input_ids", int(kwargs["input_ids"].shape[0])))
            return original_forward(*args, **kwargs)

        trainer._set_active_adapter = tracked_set_active_adapter
        trainer.model.forward = tracked_forward

        result = trainer.train()
        if result.global_step != training_args.max_steps:
            raise RuntimeError(
                f"Expected {training_args.max_steps} linear-approx LoRA EM optimizer steps, got {result.global_step}."
            )
        if not torch.isfinite(torch.tensor(result.training_loss)):
            raise RuntimeError(f"Linear-approx LoRA EM-only training loss is not finite: {result.training_loss}")

        if not mixture_forward_batches:
            raise RuntimeError("Linear-approx LoRA EM-only test did not observe mixture adapter forwards.")

        expected_anchor_rows = training_args.per_device_train_batch_size * training_args.linear_approx_num_anchors
        num_candidates = len(build_toy_listwise_dataset()[0]["responses"])
        full_candidate_rows = training_args.per_device_train_batch_size * num_candidates
        full_candidate_adapter_forwards = [
            batch for batch in mixture_forward_batches if batch[0] == "input_ids" and batch[1] >= full_candidate_rows
        ]
        oversized_anchor_forwards = [
            batch for batch in mixture_forward_batches if batch[0] == "inputs_embeds" and batch[1] > expected_anchor_rows
        ]
        if full_candidate_adapter_forwards:
            raise RuntimeError(
                "Linear-approx LoRA EM-only path used full-candidate input_ids forwards for mixture adapters: "
                f"{full_candidate_adapter_forwards}"
            )
        if oversized_anchor_forwards:
            raise RuntimeError(
                "Linear-approx LoRA EM-only path forwarded more than the selected anchors through mixture adapters: "
                f"{oversized_anchor_forwards}"
            )

    print("=" * 80)
    print("TOY MIXTURE DPO LORA EM-ONLY LINEAR APPROX SMOKE TEST")
    print("=" * 80)
    print("✓ MixtureEMDPOTrainer LoRA EM-only linear approximation completed successfully")


if __name__ == "__main__":
    run_toy_training_smoke_test()
    run_toy_lora_adapter_training_smoke_test()
    run_toy_em_only_training_smoke_test()
    run_toy_lora_em_only_training_smoke_test()
    run_toy_lora_em_only_linear_approx_smoke_test()
