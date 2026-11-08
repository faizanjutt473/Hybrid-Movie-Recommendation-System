"""Collaborative filtering: item-based KNN and (Pure) SVD matrix factorisation."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.base import BaseRecommender, as_csr
from src.config import KNN_NEIGHBORS, KNN_SHRINKAGE, RANDOM_STATE, SVD_FACTORS


class ItemKNNRecommender(BaseRecommender):
    """Item-based CF with cosine similarity.

    score(u, i) = sum over the neighbours j of i of  rating(u, j) * sim(j, i)

    Similarities are shrunk by co / (co + shrinkage), where `co` is the number of
    users who rated both items, so item pairs with little evidence count less.
    Only the `n_neighbors` most similar items are kept for every item.
    """

    name = "Item-KNN CF"

    def __init__(self, n_neighbors: int = KNN_NEIGHBORS, shrinkage: float = KNN_SHRINKAGE):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.similarity_ = None

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "ItemKNNRecommender":
        self._store_data(train, n_users, n_items)
        ratings = self.train_matrix_

        norms = np.sqrt(np.asarray(ratings.multiply(ratings).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        normalised = sparse.csr_matrix(ratings @ sparse.diags(1.0 / norms))
        similarity = (normalised.T @ normalised).toarray()

        binary = ratings.copy()
        binary.data[:] = 1.0
        co_counts = (binary.T @ binary).toarray()
        similarity *= co_counts / (co_counts + self.shrinkage)
        np.fill_diagonal(similarity, 0.0)

        k = self.n_neighbors
        if k < n_items:
            top_rows = np.argpartition(similarity, -k, axis=0)[-k:]
            keep = np.zeros(similarity.shape, dtype=bool)
            np.put_along_axis(keep, top_rows, True, axis=0)
            similarity = np.where(keep, similarity, 0.0)

        self.similarity_ = similarity
        return self

    def score_matrix(self, ratings_matrix) -> np.ndarray:
        if self.similarity_ is None:
            raise RuntimeError("Call fit() before scoring.")
        return np.asarray(as_csr(ratings_matrix) @ self.similarity_)


class SVDRecommender(BaseRecommender):
    """Latent-factor CF using truncated SVD of the user-item rating matrix.

    This is the "PureSVD" approach, which is known to work well for top-N
    ranking. A new user is folded in by projecting their rating vector on the
    item factors, so no re-training is needed:  scores = (r @ V^T) @ V.
    """

    name = "SVD CF"

    def __init__(self, n_factors: int = SVD_FACTORS, random_state: int = RANDOM_STATE):
        super().__init__()
        self.n_factors = n_factors
        self.random_state = random_state
        self.components_ = None

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "SVDRecommender":
        self._store_data(train, n_users, n_items)
        n_components = int(min(self.n_factors, min(self.train_matrix_.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        svd.fit(self.train_matrix_)
        self.components_ = svd.components_
        return self

    def score_matrix(self, ratings_matrix) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("Call fit() before scoring.")
        user_factors = np.asarray(as_csr(ratings_matrix) @ self.components_.T)
        return user_factors @ self.components_
