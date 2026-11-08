"""Top-N generation and evaluation helpers."""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.metrics import ranking_metrics


def top_n_items(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k best *unseen* items per row, best first.

    Args:
        scores: (n_users, n_items) score matrix.
        seen: boolean matrix of the same shape, True = already rated (excluded).
    """
    if k > scores.shape[1]:
        raise ValueError("k cannot exceed the number of items.")
    masked = np.where(seen, -np.inf, scores)
    candidates = np.argpartition(-masked, k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(masked, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def evaluate_scores(
    scores: np.ndarray,
    train_matrix: sparse.csr_matrix,
    relevant: Dict[int, np.ndarray],
    ks: Sequence[int],
) -> Dict[str, float]:
    """Evaluate a full score matrix against the relevant test items.

    Only users with at least one relevant test item are evaluated, and items the
    user already rated in the train set are never recommended.
    """
    if not relevant:
        raise ValueError("No users with relevant test items to evaluate.")

    users = np.array(sorted(relevant))
    n_items = scores.shape[1]

    relevant_mask = np.zeros((users.size, n_items), dtype=bool)
    for row, user in enumerate(users):
        relevant_mask[row, relevant[user]] = True

    seen = train_matrix[users].toarray() > 0
    top_items = top_n_items(scores[users], seen, max(ks))

    metrics = ranking_metrics(top_items, relevant_mask, ks, n_items)
    metrics["Users"] = int(users.size)
    return metrics


def results_table(rows: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Turn {model name: metrics} into a DataFrame with one row per model."""
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Model"
    return table
