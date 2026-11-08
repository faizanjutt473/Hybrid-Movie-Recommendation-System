"""Handling of new users (cold start) and cold-start evaluation."""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.base import as_csr
from src.config import (COLD_START_GENRE_WEIGHT, COLD_START_MIN_RATINGS,
                        COLD_START_PRIOR_STRENGTH, RELEVANCE_THRESHOLD)
from src.content_based import ContentBasedRecommender
from src.data_loader import build_user_item_matrix, get_relevant_items
from src.evaluate import evaluate_scores, top_n_items
from src.hybrid import HybridRecommender, normalize_scores
from src.popularity import PopularityRecommender


class ColdStartRecommender:
    """Serves users with little or no history.

    Strategy, based on how many ratings the user already has (n):

    * n = 0, nothing else known -> popularity (same list for everybody).
    * n = 0, favourite genres chosen at onboarding -> blend of a genre-based
      content profile and popularity.
    * 0 < n < min_ratings -> smooth blend:  conf * hybrid + (1 - conf) * fallback
      with conf = n / (n + prior_strength). The fallback acts like a prior worth
      `prior_strength` ratings, so the personalised model takes over quickly but
      gradually as evidence arrives.
    * n >= min_ratings -> the full hybrid model.
    """

    def __init__(
        self,
        hybrid: HybridRecommender,
        popularity: PopularityRecommender,
        content: ContentBasedRecommender,
        items: pd.DataFrame,
        min_ratings: int = COLD_START_MIN_RATINGS,
        genre_weight: float = COLD_START_GENRE_WEIGHT,
        prior_strength: float = COLD_START_PRIOR_STRENGTH,
    ):
        self.hybrid = hybrid
        self.popularity = popularity
        self.content = content
        self.items = items
        self.min_ratings = min_ratings
        self.genre_weight = genre_weight
        self.prior_strength = prior_strength
        self.n_items = len(items)

    def score_matrix(
        self, ratings_matrix, liked_genres: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Scores for any number of users with any amount of history."""
        ratings = as_csr(ratings_matrix)
        seen = ratings.toarray() > 0
        n_ratings = seen.sum(axis=1)

        popularity = normalize_scores(self.popularity.score_matrix(ratings), seen)
        if liked_genres:
            genre_scores = np.tile(self.content.score_from_genres(liked_genres),
                                   (ratings.shape[0], 1))
            fallback = (self.genre_weight * normalize_scores(genre_scores, seen)
                        + (1.0 - self.genre_weight) * popularity)
        else:
            fallback = popularity

        confidence = np.where(
            n_ratings >= self.min_ratings, 1.0, n_ratings / (n_ratings + self.prior_strength)
        )[:, None]
        return confidence * self.hybrid.score_matrix(ratings) + (1.0 - confidence) * fallback

    def strategy_for(self, n_ratings: int, has_genres: bool) -> str:
        if n_ratings == 0:
            return "genre onboarding + popularity" if has_genres else "popularity"
        if n_ratings < self.min_ratings:
            return "blend: hybrid + cold-start fallback"
        return "full hybrid"

    def recommend(
        self,
        user_ratings: Optional[Dict[int, float]] = None,
        liked_genres: Optional[Sequence[str]] = None,
        k: int = 10,
    ) -> pd.DataFrame:
        """Top-k recommendations for one (possibly brand-new) user.

        Args:
            user_ratings: {item_id: rating} the user has given so far (may be empty).
            liked_genres: favourite genres picked during onboarding (optional).
        """
        user_ratings = user_ratings or {}
        row = np.zeros((1, self.n_items))
        for item_id, rating in user_ratings.items():
            if not 1 <= item_id <= self.n_items:
                raise ValueError(f"Unknown item id: {item_id}")
            if not 1 <= rating <= 5:
                raise ValueError("Ratings must be between 1 and 5.")
            row[0, item_id - 1] = rating

        scores = self.score_matrix(sparse.csr_matrix(row), liked_genres)
        top = top_n_items(scores, row > 0, k)[0]

        result = self.items.iloc[top][["item_id", "title"]].copy()
        result.insert(0, "rank", np.arange(1, len(top) + 1))
        result["score"] = np.round(scores[0, top], 4)
        result["strategy"] = self.strategy_for(len(user_ratings), bool(liked_genres))
        return result.reset_index(drop=True)


def evaluate_cold_start(
    handler: ColdStartRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_users: int,
    n_items: int,
    reveal_counts: Sequence[int],
    ks: Sequence[int],
) -> pd.DataFrame:
    """Simulate new users by revealing only their first n train ratings.

    The users and test items are identical to the main evaluation, so numbers are
    comparable. Everything the user rated in train is still filtered from the
    recommendations, exactly as in the main evaluation.
    """
    train_matrix = build_user_item_matrix(train, n_users, n_items)
    relevant = get_relevant_items(test, RELEVANCE_THRESHOLD)

    ordered = train.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    position = ordered.groupby("user_id").cumcount()

    strategies = {
        "Cold-start strategy": handler.score_matrix,
        "Popularity only": handler.popularity.score_matrix,
        "Hybrid without fallback": handler.hybrid.score_matrix,
    }
    records = []
    for n_revealed in reveal_counts:
        revealed = ordered[position < n_revealed]
        revealed_matrix = build_user_item_matrix(revealed, n_users, n_items)
        for strategy, scorer in strategies.items():
            scores = scorer(revealed_matrix)
            metrics = evaluate_scores(scores, train_matrix, relevant, ks)
            records.append({"Ratings revealed": n_revealed, "Strategy": strategy, **metrics})
    return pd.DataFrame(records)
