"""Hybrid recommender: a weighted blend of popularity, CF and content scores."""

import itertools
import logging
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from src.base import BaseRecommender, as_csr
from src.config import PRIMARY_K, RELEVANCE_THRESHOLD, VAL_RATIO, WEIGHT_GRID_STEP
from src.data_loader import build_user_item_matrix, get_relevant_items, time_based_split
from src.evaluate import evaluate_scores

logger = logging.getLogger(__name__)


def normalize_scores(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Min-max scale every row to [0, 1] using only the *unseen* items.

    Models score on different scales (counts, cosine similarity, SVD values),
    so they must be put on a common scale before they can be blended. Seen items
    are ignored for the scaling because they are never recommended.
    """
    low = np.where(seen, np.inf, scores).min(axis=1, keepdims=True)
    high = np.where(seen, -np.inf, scores).max(axis=1, keepdims=True)
    span = high - low
    normalised = np.zeros_like(scores, dtype=np.float64)
    np.divide(scores - low, span, out=normalised, where=span > 0)
    return np.clip(normalised, 0.0, 1.0)


def weight_grid(names, step: float = WEIGHT_GRID_STEP) -> Iterator[Dict[str, float]]:
    """All weight combinations (multiples of `step`) that sum to 1."""
    n_steps = int(round(1.0 / step))
    for combo in itertools.product(range(n_steps + 1), repeat=len(names)):
        if sum(combo) == n_steps:
            yield {name: value / n_steps for name, value in zip(names, combo)}


class HybridRecommender(BaseRecommender):
    """Weighted hybrid of several fitted-on-the-same-data recommenders."""

    name = "Hybrid"

    def __init__(self, components: Dict[str, BaseRecommender], weights: Dict[str, float]):
        super().__init__()
        unknown = set(weights) - set(components)
        if unknown:
            raise ValueError(f"Weights given for unknown components: {unknown}")
        self.components = components
        self.weights = weights

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "HybridRecommender":
        self._store_data(train, n_users, n_items)
        for component in self.components.values():
            component.fit(train, n_users, n_items)
        return self

    def score_matrix(self, ratings_matrix) -> np.ndarray:
        ratings = as_csr(ratings_matrix)
        seen = ratings.toarray() > 0
        blended = np.zeros(seen.shape, dtype=np.float64)
        for name, weight in self.weights.items():
            if weight == 0:
                continue
            raw = self.components[name].score_matrix(ratings)
            blended += weight * normalize_scores(raw, seen)
        return blended


def tune_hybrid_weights(
    components: Dict[str, BaseRecommender],
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    k: int = PRIMARY_K,
    step: float = WEIGHT_GRID_STEP,
    val_ratio: float = VAL_RATIO,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Pick the blend weights on a validation split taken from the train data.

    The test set is never used here, so the final test numbers stay unbiased.
    Returns the best weights (by NDCG@k) and a table of every tried combination.
    """
    inner_train, validation = time_based_split(train, test_ratio=val_ratio)
    inner_matrix = build_user_item_matrix(inner_train, n_users, n_items)
    seen = inner_matrix.toarray() > 0
    relevant = get_relevant_items(validation, RELEVANCE_THRESHOLD)

    normalised = {}
    for name, component in components.items():
        component.fit(inner_train, n_users, n_items)
        normalised[name] = normalize_scores(component.score_matrix(inner_matrix), seen)

    records = []
    for weights in weight_grid(list(components), step):
        blended = sum(w * normalised[name] for name, w in weights.items() if w > 0)
        metrics = evaluate_scores(blended, inner_matrix, relevant, (k,))
        records.append({**weights, f"NDCG@{k}": metrics[f"NDCG@{k}"],
                        f"Precision@{k}": metrics[f"Precision@{k}"],
                        f"Recall@{k}": metrics[f"Recall@{k}"]})

    table = pd.DataFrame(records).sort_values(f"NDCG@{k}", ascending=False).reset_index(drop=True)
    best_row = table.iloc[0]
    best_weights = {name: float(best_row[name]) for name in components}
    logger.info("Best hybrid weights on validation: %s (NDCG@%d = %.4f)",
                best_weights, k, best_row[f"NDCG@{k}"])
    return best_weights, table
