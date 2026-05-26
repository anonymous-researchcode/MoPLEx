import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Example:
    example_id: int
    prompt: str
    responses: List[str]
    features: List[List[int]]
    ranking: List[int]
    cluster: int
    utilities: List[float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.example_id,
                "prompt": self.prompt,
                "responses": self.responses,
                "features": self.features,
                "ranking": self.ranking,
                "cluster": self.cluster,
                "utilities": self.utilities,
            },
            ensure_ascii=True,
        )


def parse_alpha(alpha_str: str, k_clusters: int) -> np.ndarray:
    if alpha_str is None:
        alpha = np.ones(k_clusters, dtype=np.float64) / k_clusters
        return alpha

    raw = np.array([float(x.strip()) for x in alpha_str.split(",")], dtype=np.float64)
    if raw.shape[0] != k_clusters:
        raise ValueError(
            f"--alpha has {raw.shape[0]} values but k_clusters={k_clusters}."
        )
    if np.any(raw < 0):
        raise ValueError("All alpha values must be non-negative.")
    if raw.sum() <= 0:
        raise ValueError("Sum of alpha must be positive.")
    return raw / raw.sum()


def format_response(item_idx: int, feats: List[int]) -> str:
    feat_text = "[ " + " , ".join(str(v) for v in feats) + " ]"
    return f"Item {item_idx} Features: {feat_text}"


def sample_gumbel(rng: np.random.Generator, shape) -> np.ndarray:
    # Gumbel(0, 1) via inverse CDF.
    u = rng.uniform(low=1e-8, high=1.0 - 1e-8, size=shape)
    return -np.log(-np.log(u))


def generate_one_example(
    example_id: int,
    rng: np.random.Generator,
    m_items: int,
    k_clusters: int,
    alpha: np.ndarray,
    value_low: int,
    value_high: int,
    prompt: str,
) -> Example:
    cluster = int(rng.choice(k_clusters, p=alpha))

    # Integer features for each response/item.
    features = rng.integers(value_low, value_high + 1, size=(m_items, k_clusters), endpoint=False)
    features = np.clip(features, value_low, value_high)

    utilities = features[:, cluster].astype(np.float64)
    noisy_scores = utilities + sample_gumbel(rng, shape=utilities.shape)
    ranking = np.argsort(-noisy_scores).tolist()

    feature_list = features.tolist()
    responses = [format_response(i, feat_row) for i, feat_row in enumerate(feature_list)]

    return Example(
        example_id=example_id,
        prompt=prompt,
        responses=responses,
        features=feature_list,
        ranking=ranking,
        cluster=cluster,
        utilities=utilities.tolist(),
    )


def write_jsonl(path: str, examples: List[Example]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic text ranking data for a mixture-of-PL task.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--m_items", type=int, default=8)
    parser.add_argument("--k_clusters", type=int, default=3)
    parser.add_argument("--value_low", type=int, default=1)
    parser.add_argument("--value_high", type=int, default=20)
    parser.add_argument("--alpha", type=str, default=None, help="Comma-separated mixture weights, e.g. '0.7,0.2,0.1'.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Rank the following items based on the user's hidden preference.",
    )
    args = parser.parse_args()

    if args.n_train <= 0 or args.n_val <= 0:
        raise ValueError("n_train and n_val must both be positive.")
    if args.m_items < 2:
        raise ValueError("m_items must be at least 2.")
    if args.k_clusters < 2:
        raise ValueError("k_clusters must be at least 2.")
    if args.value_low >= args.value_high:
        raise ValueError("value_low must be smaller than value_high.")

    alpha = parse_alpha(args.alpha, args.k_clusters)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    total = args.n_train + args.n_val
    all_examples = [
        generate_one_example(
            example_id=i,
            rng=rng,
            m_items=args.m_items,
            k_clusters=args.k_clusters,
            alpha=alpha,
            value_low=args.value_low,
            value_high=args.value_high,
            prompt=args.prompt,
        )
        for i in range(total)
    ]

    train_examples = all_examples[: args.n_train]
    val_examples = all_examples[args.n_train :]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    meta_path = os.path.join(args.output_dir, "metadata.json")

    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_train": args.n_train,
                "n_val": args.n_val,
                "m_items": args.m_items,
                "k_clusters": args.k_clusters,
                "value_low": args.value_low,
                "value_high": args.value_high,
                "alpha": alpha.tolist(),
                "seed": args.seed,
                "prompt": args.prompt,
                "format": {
                    "responses": "list[str], each string has integer feature list",
                    "features": "list[list[int]], shape (m_items, k_clusters)",
                    "ranking": "permutation of item indices in descending preference",
                    "cluster": "latent generating cluster id",
                },
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    print("Generated dataset:")
    print(f"  train: {train_path}")
    print(f"  val  : {val_path}")
    print(f"  meta : {meta_path}")


if __name__ == "__main__":
    main()
