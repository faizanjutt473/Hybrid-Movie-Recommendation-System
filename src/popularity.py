"""Popularity baseline: recommend what most people like, the same for everyone."""

import numpy as np
import pandas as pd

from src.base import BaseRecommender, as_csr
from src.config import BAYESIAN_QUANTILE


class PopularityRecommender(BaseRecommender):
    """Non-personalised baseline.

    strategy="count":    items ranked by number of ratings.
    strategy="bayesian": items ranked by a weighted (Bayesian) average rating,
                         which shrinks items with few ratings towards the global
                         mean so a single 5-star vote cannot reach the top.
    """

    def __init__(self, strategy: str = "count", quantile: float = BAYESIAN_QUANTILE):
        super().__init__()
        if strategy not in {"count", "bayesian"}:
            raise ValueError("strategy must be 'count' or 'bayesian'.")
        self.strategy = strategy
        self.quantile = quantile
        self.item_scores_ = None
        self.name = f"Popularity ({strategy})"

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "PopularityRecommender":
        self._store_data(train, n_users, n_items)
        matrix = self.train_matrix_

        counts = matrix.getnnz(axis=0).astype(np.float64)
        sums = np.asarray(matrix.sum(axis=0)).ravel()
        item_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

        if self.strategy == "count":
            # tiny mean-rating term only breaks ties between equally rated items
            self.item_scores_ = counts + 1e-3 * item_mean
        else:
            global_mean = sums.sum() / max(counts.sum(), 1.0)
            rated = counts[counts > 0]
            min_votes = float(np.quantile(rated, self.quantile)) if rated.size else 1.0
            weight = counts / (counts + min_votes)
            self.item_scores_ = weight * item_mean + (1.0 - weight) * global_mean
        return self

    def score_matrix(self, ratings_matrix) -> np.ndarray:
        if self.item_scores_ is None:
            raise RuntimeError("Call fit() before scoring.")
        n_rows = as_csr(ratings_matrix).shape[0]
        return np.tile(self.item_scores_, (n_rows, 1))
