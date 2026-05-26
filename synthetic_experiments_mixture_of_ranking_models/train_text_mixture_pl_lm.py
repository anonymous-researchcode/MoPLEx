import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset


# Compatibility shim: newer transformers may call torch.get_default_device,
# which is not available in some older torch releases.
if not hasattr(torch, "get_default_device"):
    def _fallback_get_default_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.get_default_device = _fallback_get_default_device  # type: ignore[attr-defined]
try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import wandb  # type: ignore

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RankingDataset(Dataset):
    def __init__(self, rows: List[dict]):
        if len(rows) == 0:
            raise ValueError("Empty dataset.")
        self.rows = rows
        m_items = len(rows[0]["responses"])
        k_clusters = len(rows[0]["features"][0])
        for r in rows:
            if len(r["responses"]) != m_items:
                raise ValueError("All examples must share same m_items.")
            if len(r["ranking"]) != m_items:
                raise ValueError("All rankings must be full permutations.")
            if len(r["features"][0]) != k_clusters:
                raise ValueError("All examples must share same k_clusters/features length.")
        self.m_items = m_items
        self.k_clusters = k_clusters

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        return {
            "prompt": r["prompt"],
            "responses": r["responses"],
            "features": torch.tensor(r["features"], dtype=torch.float32),
            "ranking": torch.tensor(r["ranking"], dtype=torch.long),
            "cluster": torch.tensor(r["cluster"], dtype=torch.long),
        }


TOKEN_RE = re.compile(r"\[|\]|,|:|-?\d+|[A-Za-z_]+")


class SimpleTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id: Dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.id_to_token = [self.PAD, self.UNK]

    def tokenize(self, text: str) -> List[str]:
        return TOKEN_RE.findall(text)

    def fit(self, texts: List[str]) -> None:
        counts: Dict[str, int] = {}
        for t in texts:
            for tok in self.tokenize(t):
                counts[tok] = counts.get(tok, 0) + 1
        sorted_tokens = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        for tok, _ in sorted_tokens:
            if tok in self.token_to_id:
                continue
            if len(self.id_to_token) >= self.max_vocab_size:
                break
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, max_length: int) -> Tuple[List[int], List[int]]:
        token_ids = [self.token_to_id.get(tok, 1) for tok in self.tokenize(text)]
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        attention = [1] * len(token_ids)
        pad_len = max_length - len(token_ids)
        if pad_len > 0:
            token_ids.extend([0] * pad_len)
            attention.extend([0] * pad_len)
        return token_ids, attention

    def batch_encode(self, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
        all_ids, all_attn = [], []
        for t in texts:
            ids, attn = self.encode(t, max_length=max_length)
            all_ids.append(ids)
            all_attn.append(attn)
        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(all_attn, dtype=torch.long, device=device),
        }


class TinyLMBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        pad_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled


class HFBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled


class MixturePLModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, k_clusters: int, use_router: bool):
        super().__init__()
        self.backbone = backbone
        self.use_router = use_router
        self.k_clusters = k_clusters
        self.reward_heads = nn.Linear(hidden_size, k_clusters, bias=True)
        if self.use_router:
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, k_clusters),
            )
            self.global_router_logits = None
        else:
            self.router = None
            self.global_router_logits = nn.Parameter(torch.zeros(k_clusters))

    def encode_items(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, m_items: int) -> torch.Tensor:
        bsz = input_ids.shape[0] // m_items
        item_repr = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return item_repr.view(bsz, m_items, -1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, m_items: int) -> Dict[str, torch.Tensor]:
        item_repr = self.encode_items(input_ids=input_ids, attention_mask=attention_mask, m_items=m_items)
        # rewards[b, m, k] -> transpose to rewards[b, k, m]
        rewards = self.reward_heads(item_repr).transpose(1, 2)
        context = item_repr.mean(dim=1)
        if self.use_router:
            router_logits = self.router(context)
        else:
            bsz = context.shape[0]
            router_logits = self.global_router_logits.unsqueeze(0).expand(bsz, -1)
        return {
            "rewards": rewards,
            "router_logits": router_logits,
            "item_repr": item_repr,
        }


def pl_log_prob(rewards: torch.Tensor, rankings: torch.Tensor) -> torch.Tensor:
    # rewards: [B, M], rankings: [B, M]
    ranked = rewards.gather(1, rankings)
    out = torch.zeros(rewards.shape[0], device=rewards.device)
    m_items = rewards.shape[1]
    for t in range(m_items - 1):
        logits = ranked[:, t:]
        out = out + logits[:, 0] - torch.logsumexp(logits, dim=1)
    return out


def mixture_pl_nll(router_logits: torch.Tensor, rewards: torch.Tensor, rankings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # router_logits: [B, K], rewards: [B, K, M]
    log_alpha = F.log_softmax(router_logits, dim=1)
    bsz, k_clusters, _ = rewards.shape
    comp_logp = []
    for c in range(k_clusters):
        comp_logp.append(pl_log_prob(rewards[:, c, :], rankings))
    comp_logp = torch.stack(comp_logp, dim=1)
    mix_logp = torch.logsumexp(log_alpha + comp_logp, dim=1)
    nll = -mix_logp.mean()
    return nll, comp_logp


def em_responsibilities(router_logits: torch.Tensor, comp_logp: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # gamma[b, c] = p(z=c | ranking, context) under current parameters.
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
) -> torch.Tensor:
    # Generalized M-step objective: maximize E_q[log p(z, pi | x)].
    log_alpha = F.log_softmax(router_logits, dim=1)
    expected_joint = (gamma * (log_alpha + comp_logp)).sum(dim=1)
    return -expected_joint.mean()


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


@dataclass
class EvalMetrics:
    nll: float
    router_acc: float
    posterior_acc_raw: float
    posterior_acc_aligned: float
    aligned_mean_corr: float
    assignment: List[int]


@torch.no_grad()
def evaluate(
    model: MixturePLModel,
    dataloader: DataLoader,
    device: torch.device,
    encode_batch,
    k_clusters: int,
) -> EvalMetrics:
    model.eval()
    total_nll = 0.0
    total = 0
    router_correct = 0
    posterior_correct_raw = 0

    # For head-feature correlation matrix.
    all_rewards = [[] for _ in range(k_clusters)]
    all_features = [[] for _ in range(k_clusters)]
    all_true_clusters = []
    all_posterior_pred = []

    for batch in dataloader:
        rankings = batch["ranking"].to(device)
        clusters = batch["cluster"].to(device)
        features = batch["features"].cpu().numpy()  # [B, M, K]
        m_items = rankings.shape[1]

        texts = []
        for prompt, responses in zip(batch["prompt"], batch["responses"]):
            for resp in responses:
                texts.append(f"Prompt: {prompt}\nResponse: {resp}")

        enc = encode_batch(texts, device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], m_items=m_items)
        nll, comp_logp = mixture_pl_nll(out["router_logits"], out["rewards"], rankings)

        bsz = rankings.shape[0]
        total_nll += float(nll.item()) * bsz
        total += bsz

        pred_cluster = out["router_logits"].argmax(dim=1)
        router_correct += int((pred_cluster == clusters).sum().item())

        # Posterior cluster prediction via argmax_c log p(z=c, pi|x).
        posterior_logits = F.log_softmax(out["router_logits"], dim=1) + comp_logp
        posterior_pred = posterior_logits.argmax(dim=1)
        posterior_correct_raw += int((posterior_pred == clusters).sum().item())
        all_true_clusters.append(clusters.detach().cpu().numpy())
        all_posterior_pred.append(posterior_pred.detach().cpu().numpy())

        rewards = out["rewards"].detach().cpu().numpy()  # [B, K_head, M]
        for head_idx in range(k_clusters):
            all_rewards[head_idx].append(rewards[:, head_idx, :].reshape(-1))
        for feat_idx in range(k_clusters):
            all_features[feat_idx].append(features[:, :, feat_idx].reshape(-1))

    corr_mat = np.zeros((k_clusters, k_clusters), dtype=np.float64)
    for h in range(k_clusters):
        rvec = np.concatenate(all_rewards[h], axis=0)
        for f in range(k_clusters):
            fvec = np.concatenate(all_features[f], axis=0)
            corr_mat[h, f] = pearson_corr(rvec, fvec)

    # Maximize correlation under permutation matching.
    row_ind, col_ind = linear_sum_assignment(-corr_mat)
    aligned_corr = corr_mat[row_ind, col_ind]

    pred_all = np.concatenate(all_posterior_pred, axis=0)
    true_all = np.concatenate(all_true_clusters, axis=0)
    mapped_pred = np.array([col_ind[p] for p in pred_all], dtype=np.int64)
    posterior_acc_aligned = float((mapped_pred == true_all).mean())

    return EvalMetrics(
        nll=total_nll / max(total, 1),
        router_acc=router_correct / max(total, 1),
        posterior_acc_raw=posterior_correct_raw / max(total, 1),
        posterior_acc_aligned=posterior_acc_aligned,
        aligned_mean_corr=float(aligned_corr.mean()),
        assignment=col_ind.tolist(),
    )


def build_collate_fn() -> callable:
    def collate(batch: List[dict]) -> dict:
        return {
            "prompt": [x["prompt"] for x in batch],
            "responses": [x["responses"] for x in batch],
            "features": torch.stack([x["features"] for x in batch], dim=0),
            "ranking": torch.stack([x["ranking"] for x in batch], dim=0),
            "cluster": torch.stack([x["cluster"] for x in batch], dim=0),
        }

    return collate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a text-backbone Mixture-of-Plackett-Luce model.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=0,
        help="If >0 and smaller than the full training set, randomly subsample this many training examples.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--em_temperature", type=float, default=1.0)
    parser.add_argument(
        "--m_step_updates",
        type=int,
        default=10,
        help="Number of gradient updates to run in each M-step using fixed E-step responsibilities.",
    )
    parser.add_argument(
        "--log_every_batches",
        type=int,
        default=0,
        help="If >0, print running train metrics every N batches within each epoch.",
    )
    parser.add_argument(
        "--eval_every_batches",
        type=int,
        default=0,
        help="If >0, run validation every N batches within each epoch (can be expensive).",
    )
    parser.add_argument(
        "--use_router",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, use contextual router alpha(x). If false, learn global mixture weights alpha.",
    )

    # Tiny backbone args.
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_vocab_size", type=int, default=5000)

    # HF backbone args.
    parser.add_argument("--hf_model_name", type=str, default=None)
    parser.add_argument("--freeze_hf_backbone", action="store_true")
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="anonymous",
        help="WandB entity (organization or username).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="multimodal-preference-optimization",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional WandB run name.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="WandB mode.",
    )
    args = parser.parse_args()

    if args.m_step_updates < 1:
        raise ValueError("m_step_updates must be at least 1.")
    if args.train_subset_size < 0:
        raise ValueError("train_subset_size must be >= 0.")
    if args.log_every_batches < 0:
        raise ValueError("log_every_batches must be >= 0.")
    if args.eval_every_batches < 0:
        raise ValueError("eval_every_batches must be >= 0.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb and not HAS_WANDB:
        raise ImportError("wandb is not installed. Install it or pass --no-use_wandb.")

    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=vars(args),
            dir=args.output_dir,
        )

    train_rows = read_jsonl(args.train_path)
    val_rows = read_jsonl(args.val_path)

    full_train_size = len(train_rows)
    if args.train_subset_size > 0 and args.train_subset_size < full_train_size:
        rng = random.Random(args.seed)
        chosen_idx = rng.sample(range(full_train_size), args.train_subset_size)
        train_rows = [train_rows[i] for i in chosen_idx]
        print(
            f"Using random train subset: {len(train_rows)} / {full_train_size} "
            f"examples (seed={args.seed})."
        )
    else:
        print(f"Using full train set: {full_train_size} examples.")

    train_ds = RankingDataset(train_rows)
    val_ds = RankingDataset(val_rows)

    if train_ds.m_items != val_ds.m_items or train_ds.k_clusters != val_ds.k_clusters:
        raise ValueError("Train/val have incompatible m_items or k_clusters.")

    m_items = train_ds.m_items
    k_clusters = train_ds.k_clusters

    collate_fn = build_collate_fn()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Build encoding stack.
    if args.hf_model_name is not None:
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is not installed, but --hf_model_name was provided. Install transformers or use tiny backbone."
            )
        hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        backbone = HFBackbone(args.hf_model_name)
        hidden_size = backbone.hidden_size

        def encode_batch(texts: List[str], dev: torch.device) -> Dict[str, torch.Tensor]:
            enc = hf_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            return {k: v.to(dev) for k, v in enc.items()}

    else:
        tokenizer = SimpleTokenizer(max_vocab_size=args.max_vocab_size)
        train_texts = []
        for r in train_rows:
            for resp in r["responses"]:
                train_texts.append(f"Prompt: {r['prompt']}\nResponse: {resp}")
        tokenizer.fit(train_texts)

        backbone = TinyLMBackbone(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_len=args.max_length,
            dropout=args.dropout,
        )
        hidden_size = args.d_model

        def encode_batch(texts: List[str], dev: torch.device) -> Dict[str, torch.Tensor]:
            return tokenizer.batch_encode(texts=texts, max_length=args.max_length, device=dev)

    model = MixturePLModel(
        backbone=backbone,
        hidden_size=hidden_size,
        k_clusters=k_clusters,
        use_router=args.use_router,
    )
    model.to(device)

    if args.hf_model_name is not None and args.freeze_hf_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = math.inf
    best_path = os.path.join(args.output_dir, "best_model.pt")
    history_path = os.path.join(args.output_dir, "history.jsonl")

    with open(history_path, "w", encoding="utf-8") as _:
        pass

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_nll = 0.0
        running_entropy = 0.0
        running_batches = 0
        seen = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            global_step += 1
            rankings = batch["ranking"].to(device)

            texts = []
            for prompt, responses in zip(batch["prompt"], batch["responses"]):
                for resp in responses:
                    texts.append(f"Prompt: {prompt}\nResponse: {resp}")

            # E-step: infer soft latent assignments with current parameters.
            enc = encode_batch(texts, device)
            with torch.no_grad():
                out_e = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], m_items=m_items)
                _, comp_logp_e = mixture_pl_nll(out_e["router_logits"], out_e["rewards"], rankings)
                gamma = em_responsibilities(
                    router_logits=out_e["router_logits"],
                    comp_logp=comp_logp_e,
                    temperature=args.em_temperature,
                )

            # M-step: optimize expected complete-data log-likelihood using fixed gamma.
            # We can take multiple gradient steps per E-step (generalized EM).
            last_loss = None
            last_nll = None
            for _ in range(args.m_step_updates):
                out_m = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], m_items=m_items)
                nll, comp_logp_m = mixture_pl_nll(out_m["router_logits"], out_m["rewards"], rankings)
                loss = em_expected_complete_nll(out_m["router_logits"], comp_logp_m, gamma)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                last_loss = loss
                last_nll = nll

            bsz = rankings.shape[0]
            running_loss += float(last_loss.item()) * bsz
            running_nll += float(last_nll.item()) * bsz
            gamma_entropy = -(gamma * torch.log(gamma.clamp_min(1e-12))).sum(dim=1).mean()
            running_entropy += float(gamma_entropy.item()) * bsz
            running_batches += 1
            seen += bsz

            if args.log_every_batches > 0 and (batch_idx % args.log_every_batches == 0):
                elapsed = time.time() - epoch_start
                batches_total = len(train_loader)
                progress = 100.0 * batch_idx / max(batches_total, 1)
                avg_em_loss = running_loss / max(seen, 1)
                avg_nll = running_nll / max(seen, 1)
                avg_entropy = running_entropy / max(seen, 1)
                print(
                    f"Epoch {epoch:02d} [{batch_idx}/{batches_total} {progress:.1f}%] | "
                    f"train_em_loss={avg_em_loss:.4f} | "
                    f"train_nll={avg_nll:.4f} | "
                    f"gamma_H={avg_entropy:.4f} | "
                    f"elapsed={elapsed:.1f}s"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/epoch": epoch,
                            "train/epoch_progress": progress / 100.0,
                            "train/em_loss_running": avg_em_loss,
                            "train/nll_running": avg_nll,
                            "train/gamma_entropy_running": avg_entropy,
                            "train/elapsed_sec": elapsed,
                        },
                        step=global_step,
                    )

            if args.eval_every_batches > 0 and (batch_idx % args.eval_every_batches == 0):
                mid_metrics = evaluate(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    encode_batch=encode_batch,
                    k_clusters=k_clusters,
                )
                print(
                    f"Epoch {epoch:02d} MidEval @batch {batch_idx}/{len(train_loader)} | "
                    f"val_nll={mid_metrics.nll:.4f} | "
                    f"router_acc={mid_metrics.router_acc:.4f} | "
                    f"post_acc_raw={mid_metrics.posterior_acc_raw:.4f} | "
                    f"post_acc_aligned={mid_metrics.posterior_acc_aligned:.4f} | "
                    f"aligned_corr={mid_metrics.aligned_mean_corr:.4f}"
                )
                if wandb_run is not None:
                    wandb.log(
                        {
                            "val_mid/nll": mid_metrics.nll,
                            "val_mid/router_acc": mid_metrics.router_acc,
                            "val_mid/posterior_acc_raw": mid_metrics.posterior_acc_raw,
                            "val_mid/posterior_acc_aligned": mid_metrics.posterior_acc_aligned,
                            "val_mid/aligned_mean_corr": mid_metrics.aligned_mean_corr,
                        },
                        step=global_step,
                    )
                model.train()

        train_loss = running_loss / max(seen, 1)
        train_nll = running_nll / max(seen, 1)
        train_gamma_entropy = running_entropy / max(seen, 1)
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            encode_batch=encode_batch,
            k_clusters=k_clusters,
        )

        record = {
            "epoch": epoch,
            "train_em_loss": train_loss,
            "train_nll": train_nll,
            "train_gamma_entropy": train_gamma_entropy,
            "val_nll": val_metrics.nll,
            "val_router_acc": val_metrics.router_acc,
            "val_posterior_acc_raw": val_metrics.posterior_acc_raw,
            "val_posterior_acc_aligned": val_metrics.posterior_acc_aligned,
            "val_aligned_mean_corr": val_metrics.aligned_mean_corr,
            "alignment": val_metrics.assignment,
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

        if wandb_run is not None:
            wandb.log(
                {
                    "val/nll": val_metrics.nll,
                    "val/router_acc": val_metrics.router_acc,
                    "val/posterior_acc_raw": val_metrics.posterior_acc_raw,
                    "val/posterior_acc_aligned": val_metrics.posterior_acc_aligned,
                    "val/aligned_mean_corr": val_metrics.aligned_mean_corr,
                    "train/em_loss_epoch": train_loss,
                    "train/nll_epoch": train_nll,
                    "train/gamma_entropy_epoch": train_gamma_entropy,
                    "train/epoch": epoch,
                },
                step=global_step,
            )

        print(
            f"Epoch {epoch:02d} | train_em_loss={train_loss:.4f} | "
            f"train_nll={train_nll:.4f} | "
            f"gamma_H={train_gamma_entropy:.4f} | "
            f"val_nll={val_metrics.nll:.4f} | "
            f"router_acc={val_metrics.router_acc:.4f} | "
            f"post_acc_raw={val_metrics.posterior_acc_raw:.4f} | "
            f"post_acc_aligned={val_metrics.posterior_acc_aligned:.4f} | "
            f"aligned_corr={val_metrics.aligned_mean_corr:.4f} | "
            f"align={val_metrics.assignment}"
        )

        if val_metrics.nll < best_val:
            best_val = val_metrics.nll
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "k_clusters": k_clusters,
                    "m_items": m_items,
                    "args": vars(args),
                },
                best_path,
            )

    print("Training complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"History log: {history_path}")
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
