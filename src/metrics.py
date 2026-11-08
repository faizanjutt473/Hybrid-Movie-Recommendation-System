"""Vectorised ranking metrics: Precision@K, Recall@K, NDCG@K, HitRate@K, Coverage@K."""

from typing import Dict, Sequence

import numpy as np


def ranking_metrics(
    top_items: np.ndarray,
    relevant_mask: np.ndarray,
    ks: Sequence[int],
    n_items: int,
) -> Dict[str, float]:
    """Average ranking metrics over users.

    Args:
        top_items: (n_users, >= max(ks)) item indices, best first.
        relevant_mask: (n_users, n_items) boolean matrix, True = relevant item.
        ks: cut-offs, e.g. (5, 10, 20).
        n_items: catalogue size (used for coverage).

    Binary relevance is used for NDCG: gain 1 for a relevant item, 0 otherwise.
    """
    k_max = max(ks)
    if top_items.shape[1] < k_max:
        raise ValueError("top_items must contain at least max(ks) columns.")

    hits = np.take_along_axis(relevant_mask, top_items[:, :k_max], axis=1)
    n_relevant = relevant_mask.sum(axis=1)

    discounts = 1.0 / np.log2(np.arange(2, k_max + 2))
    ideal_cumulative = np.cumsum(discounts)

    results: Dict[str, float] = {}
    for k in ks:
        hits_k = hits[:, :k]
        n_hits = hits_k.sum(axis=1)

        precision = n_hits / k
        recall = n_hits / n_relevant
        dcg = (hits_k * discounts[:k]).sum(axis=1)
        ideal_dcg = ideal_cumulative[np.minimum(n_relevant, k) - 1]

        results[f"Precision@{k}"] = float(precision.mean())
        results[f"Recall@{k}"] = float(recall.mean())
        results[f"NDCG@{k}"] = float((dcg / ideal_dcg).mean())
        results[f"HitRate@{k}"] = float((n_hits > 0).mean())
        results[f"Coverage@{k}"] = float(np.unique(top_items[:, :k]).size / n_items)
    return results
