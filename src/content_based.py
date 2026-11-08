"""Content-based recommender built from movie genres and release decade."""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from src.base import BaseRecommender, as_csr
from src.config import LIKE_THRESHOLD
from src.data_loader import GENRES


class ContentBasedRecommender(BaseRecommender):
    """Recommend movies whose content resembles what the user liked.

    * Item vector: genre flags + release decade, re-weighted with TF-IDF so rare
      genres (e.g. Film-Noir) count more than very common ones (e.g. Drama).
    * User profile: rating-weighted sum of the vectors of the movies the user
      liked (rating >= like_threshold); users who liked nothing fall back to all
      of their rated movies.
    * Score: cosine similarity between the user profile and every movie.
    """

    name = "Content-Based"

    def __init__(self, items: pd.DataFrame, like_threshold: float = LIKE_THRESHOLD):
        super().__init__()
        self.items = items
        self.like_threshold = like_threshold
        self.tfidf_ = None
        self.item_features_ = None

    def _raw_features(self) -> np.ndarray:
        genre_flags = self.items[GENRES].to_numpy(dtype=np.float64)
        decade = (self.items["release_year"] // 10 * 10).fillna(-1).astype(int)
        decade_flags = pd.get_dummies(decade).to_numpy(dtype=np.float64)
        return np.hstack([genre_flags, decade_flags])

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int) -> "ContentBasedRecommender":
        self._store_data(train, n_users, n_items)
        if len(self.items) != n_items:
            raise ValueError("Items table and n_items do not match.")
        self.tfidf_ = TfidfTransformer(norm="l2")
        self.item_features_ = sparse.csr_matrix(
            self.tfidf_.fit_transform(self._raw_features())
        )
        return self

    def _preference_weights(self, ratings: sparse.csr_matrix) -> sparse.csr_matrix:
        """Keep liked ratings; fall back to all ratings if a user liked nothing."""
        liked = ratings.copy()
        liked.data[liked.data < self.like_threshold] = 0.0
        liked.eliminate_zeros()

        has_liked = np.diff(liked.indptr) > 0
        has_rated = np.diff(ratings.indptr) > 0
        fallback = (~has_liked & has_rated).astype(np.float64)
        return sparse.csr_matrix(liked + sparse.diags(fallback) @ ratings)

    def score_matrix(self, ratings_matrix) -> np.ndarray:
        if self.item_features_ is None:
            raise RuntimeError("Call fit() before scoring.")
        weights = self._preference_weights(as_csr(ratings_matrix))
        profiles = normalize(weights @ self.item_features_)
        return (profiles @ self.item_features_.T).toarray()

    def score_from_genres(self, genres: Sequence[str]) -> np.ndarray:
        """Score every item for a brand-new user who only picked favourite genres."""
        if self.item_features_ is None:
            raise RuntimeError("Call fit() before scoring.")
        unknown = [g for g in genres if g not in GENRES]
        if unknown:
            raise ValueError(f"Unknown genres: {unknown}. Valid genres: {GENRES}")
        vector = np.zeros((1, self.item_features_.shape[1]))
        for genre in genres:
            vector[0, GENRES.index(genre)] = 1.0
        profile = normalize(sparse.csr_matrix(self.tfidf_.transform(vector)))
        return (self.item_features_ @ profile.T).toarray().ravel()
