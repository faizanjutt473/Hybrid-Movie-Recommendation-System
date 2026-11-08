"""Common interface shared by every recommender."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import sparse

from src.data_loader import build_user_item_matrix


def as_csr(matrix) -> sparse.csr_matrix:
    """Return `matrix` as a float64 CSR matrix."""
    return sparse.csr_matrix(matrix, dtype=np.float64)


class BaseRecommender(ABC):
    """Every model learns from train ratings and scores *all* items for users.

    `score_matrix` receives a (users x items) rating matrix and returns a dense
    (users x items) score matrix. Because it works on arbitrary rating rows it
    can score existing users as well as brand-new users ("fold-in"), which is
    what the cold-start logic relies on.
    """

    name = "base"

    def __init__(self) -> None:
        self.n_users = 0
        self.n_items = 0
        self.train_matrix_ = None

    def _store_data(self, train: pd.DataFrame, n_users: int, n_items: int) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.train_matrix_ = build_user_item_matrix(train, n_users, n_items)

    @abstractmethod
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "BaseRecommender":
        """Learn from the train ratings."""

    @abstractmethod
    def score_matrix(self, ratings_matrix) -> np.ndarray:
        """Score every item for every row of `ratings_matrix`."""

    def predict_scores(self) -> np.ndarray:
        """Scores for all training users."""
        return self.score_matrix(self.train_matrix_)
