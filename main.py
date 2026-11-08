"""End-to-end pipeline: split, train, evaluate and report all recommenders.

Run from the project root:
    python main.py
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src import config
from src.cold_start import ColdStartRecommender, evaluate_cold_start
from src.collaborative import ItemKNNRecommender, SVDRecommender
from src.content_based import ContentBasedRecommender
from src.data_loader import (build_user_item_matrix, get_dimensions,
                             get_relevant_items, load_items, load_ratings,
                             time_based_split, verify_no_leakage)
from src.evaluate import evaluate_scores, results_table, top_n_items
from src.hybrid import HybridRecommender, tune_hybrid_weights
from src.popularity import PopularityRecommender
from src.visualize import plot_cold_start, plot_model_comparison

logger = logging.getLogger("recsys")


def build_components(items: pd.DataFrame) -> dict:
    """Fresh (unfitted) building blocks used by the hybrid model."""
    return {
        "popularity": PopularityRecommender("count"),
        "item_knn": ItemKNNRecommender(),
        "svd": SVDRecommender(),
        "content": ContentBasedRecommender(items),
    }


def write_sample_recommendations(path: Path, hybrid, handler, items, train, n_users, n_items, k):
    """Save readable examples: a warm user and three kinds of new user."""
    train_matrix = build_user_item_matrix(train, n_users, n_items)
    lines = []

    # 1) a returning user with plenty of history
    activity = train.groupby("user_id").size()
    user_id = int(activity.idxmax())
    history = (train[train["user_id"] == user_id].sort_values("rating", ascending=False)
               .head(5).merge(items[["item_id", "title"]], on="item_id"))
    scores = hybrid.score_matrix(train_matrix[user_id - 1])
    top = top_n_items(scores, train_matrix[user_id - 1].toarray() > 0, k)[0]
    lines.append(f"=== Existing user {user_id} ({activity[user_id]} train ratings) ===")
    lines.append("Highest rated movies in history: " + "; ".join(history["title"]))
    lines.append("Hybrid recommendations:")
    lines += [f"  {r}. {items.loc[i, 'title']}" for r, i in enumerate(top, start=1)]

    # 2) brand-new user, no history at all
    lines.append("\n=== New user: no history, no preferences ===")
    result = handler.recommend(k=k)
    lines.append(f"Strategy: {result['strategy'].iloc[0]}")
    lines += [f"  {r.rank}. {r.title}" for r in result.itertuples()]

    # 3) brand-new user who picked favourite genres during onboarding
    genres = ["Action", "Sci-Fi"]
    lines.append(f"\n=== New user: onboarding genres {genres} ===")
    result = handler.recommend(liked_genres=genres, k=k)
    lines.append(f"Strategy: {result['strategy'].iloc[0]}")
    lines += [f"  {r.rank}. {r.title}" for r in result.itertuples()]

    # 4) new user who has rated three popular Sci-Fi movies
    popularity = handler.popularity.item_scores_
    sci_fi = items[items["Sci-Fi"] == 1]["item_id"].to_numpy()
    liked = sci_fi[popularity[sci_fi - 1].argsort()[::-1][:3]]
    ratings = {int(i): 5.0 for i in liked}
    lines.append("\n=== New user: 3 ratings (5 stars) for "
                 + "; ".join(items.loc[items["item_id"].isin(liked), "title"]) + " ===")
    result = handler.recommend(user_ratings=ratings, k=k)
    lines.append(f"Strategy: {result['strategy'].iloc[0]}")
    lines += [f"  {r.rank}. {r.title}" for r in result.itertuples()]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(data_dir: Path, results_dir: Path, skip_tuning: bool) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    ks = config.TOP_K_VALUES
    k_main = config.PRIMARY_K

    # ----------------------------------------------------------------- data --
    ratings = load_ratings(data_dir)
    items = load_items(data_dir)
    n_users, n_items = get_dimensions(ratings, items)
    train, test = time_based_split(ratings, config.TEST_RATIO, config.MIN_USER_RATINGS)
    verify_no_leakage(train, test)

    train_matrix = build_user_item_matrix(train, n_users, n_items)
    relevant = get_relevant_items(test, config.RELEVANCE_THRESHOLD)
    logger.info("Users: %d | Items: %d | Ratings: %d", n_users, n_items, len(ratings))
    logger.info("Train: %d | Test: %d | Users with relevant test items: %d",
                len(train), len(test), len(relevant))

    # ------------------------------------------- baselines and single models --
    models = {
        "Popularity (count)": PopularityRecommender("count"),
        "Popularity (bayesian)": PopularityRecommender("bayesian"),
        "Item-KNN CF": ItemKNNRecommender(),
        "SVD CF": SVDRecommender(),
        "Content-Based": ContentBasedRecommender(items),
    }
    rows = {}
    for name, model in models.items():
        model.fit(train, n_users, n_items)
        rows[name] = evaluate_scores(model.predict_scores(), train_matrix, relevant, ks)
        logger.info("%-22s NDCG@%d = %.4f", name, k_main, rows[name][f"NDCG@{k_main}"])

    # ---------------------------------------------------------------- hybrid --
    if skip_tuning:
        weights = {"popularity": 0.1, "item_knn": 0.3, "svd": 0.4, "content": 0.2}
        logger.info("Tuning skipped, using default weights %s", weights)
    else:
        weights, tuning_table = tune_hybrid_weights(
            build_components(items), train, n_users, n_items, k=k_main,
            step=config.WEIGHT_GRID_STEP, val_ratio=config.VAL_RATIO)
        tuning_table.head(20).round(4).to_csv(results_dir / "hybrid_weight_search.csv", index=False)

    hybrid = HybridRecommender(build_components(items), weights)
    hybrid.fit(train, n_users, n_items)
    rows["Hybrid"] = evaluate_scores(hybrid.predict_scores(), train_matrix, relevant, ks)
    logger.info("%-22s NDCG@%d = %.4f", "Hybrid", k_main, rows["Hybrid"][f"NDCG@{k_main}"])

    table = results_table(rows)
    table.round(4).to_csv(results_dir / "metrics.csv")
    (results_dir / "hybrid_weights.json").write_text(json.dumps(weights, indent=2))
    plot_model_comparison(table, k_main, results_dir / "model_comparison.png")

    # ------------------------------------------------------------ cold start --
    handler = ColdStartRecommender(
        hybrid, hybrid.components["popularity"], hybrid.components["content"], items)
    cold_table = evaluate_cold_start(handler, train, test, n_users, n_items,
                                     config.COLD_START_REVEAL, ks)
    cold_table.round(4).to_csv(results_dir / "cold_start_metrics.csv", index=False)
    plot_cold_start(cold_table, k_main, results_dir / "cold_start.png")

    write_sample_recommendations(results_dir / "sample_recommendations.txt",
                                 hybrid, handler, items, train, n_users, n_items, k_main)

    # --------------------------------------------------------------- summary --
    show = [f"Precision@{k_main}", f"Recall@{k_main}", f"NDCG@{k_main}",
            f"HitRate@{k_main}", f"Coverage@{k_main}"]
    print(f"\nTest-set results at K = {k_main} ({table['Users'].iloc[0]} users)")
    print(table[show].round(4).to_string())
    print("\nCold-start (NDCG@%d):" % k_main)
    print(cold_table.pivot(index="Ratings revealed", columns="Strategy",
                           values=f"NDCG@{k_main}").round(4).to_string())
    print(f"\nAll files saved in: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens recommendation system")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--skip-tuning", action="store_true",
                        help="use default hybrid weights instead of the validation search")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    main(args.data_dir, args.results_dir, args.skip_tuning)
