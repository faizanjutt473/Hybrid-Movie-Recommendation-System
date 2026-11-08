"""Data loading, time-based splitting and user-item matrix utilities."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load ratings with columns: user_id, item_id, rating, timestamp."""
    path = Path(data_dir) / "u.data"
    if not path.exists():
        raise FileNotFoundError(f"Ratings file not found: {path}")
    ratings = pd.read_csv(
        path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    # a (user, item) pair must be unique, otherwise the sparse matrix would sum it
    ratings = ratings.drop_duplicates(["user_id", "item_id"], keep="last")
    return ratings.reset_index(drop=True)


def load_items(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load movie metadata (title, release year and the 19 genre flags)."""
    path = Path(data_dir) / "u.item"
    if not path.exists():
        raise FileNotFoundError(f"Items file not found: {path}")
    columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
    items = pd.read_csv(path, sep="|", names=columns, encoding="latin-1")
    items["release_year"] = pd.to_datetime(
        items["release_date"], format="%d-%b-%Y", errors="coerce"
    ).dt.year
    items = items.drop(columns=["release_date", "video_release_date", "imdb_url"])
    items = items.sort_values("item_id").reset_index(drop=True)

    # the models index items by (item_id - 1), so ids must be 1..n without gaps
    if not np.array_equal(items["item_id"].to_numpy(), np.arange(1, len(items) + 1)):
        raise ValueError("item_id values must be contiguous and start at 1.")
    return items


def get_dimensions(ratings: pd.DataFrame, items: pd.DataFrame) -> Tuple[int, int]:
    """Return (n_users, n_items). User ids are expected to be 1..n_users."""
    n_users = int(ratings["user_id"].max())
    n_items = len(items)
    if ratings["item_id"].max() > n_items:
        raise ValueError("Ratings reference item ids that are missing in the items file.")
    return n_users, n_items


def time_based_split(
    ratings: pd.DataFrame, test_ratio: float = 0.2, min_ratings: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split.

    For every user the oldest (1 - test_ratio) of the ratings go to the train
    set and the newest test_ratio go to the test set, so the model never sees
    the "future". Users with fewer than `min_ratings` ratings are kept entirely
    in the train set.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    df = ratings.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    ).copy()
    position = df.groupby("user_id").cumcount()
    count = df.groupby("user_id")["item_id"].transform("count")

    first_test_position = np.floor(count * (1 - test_ratio)).astype(int)
    is_test = (position >= first_test_position) & (count >= min_ratings)

    train = df[~is_test].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    return train, test


def verify_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if, for any user, a train rating is newer than a test rating."""
    last_train = train.groupby("user_id")["timestamp"].max()
    first_test = test.groupby("user_id")["timestamp"].min()
    joined = pd.concat([last_train, first_test], axis=1, keys=["train", "test"]).dropna()
    if (joined["train"] > joined["test"]).any():
        raise AssertionError("Temporal leakage detected between train and test.")


def build_user_item_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    """Sparse user x item matrix holding ratings (0 = not rated)."""
    return csr_matrix(
        (
            df["rating"].to_numpy(dtype=np.float64),
            (df["user_id"].to_numpy() - 1, df["item_id"].to_numpy() - 1),
        ),
        shape=(n_users, n_items),
    )


def get_relevant_items(test: pd.DataFrame, threshold: float) -> Dict[int, np.ndarray]:
    """Map user index -> array of item indices the user rated >= threshold."""
    relevant = test[test["rating"] >= threshold]
    return {
        int(user) - 1: group["item_id"].to_numpy() - 1
        for user, group in relevant.groupby("user_id")
    }
