"""Plots for the evaluation results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, we only save files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_model_comparison(table: pd.DataFrame, k: int, path: Path) -> None:
    """Grouped bars: Precision@k, Recall@k and NDCG@k for every model."""
    metrics = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"]
    x = np.arange(len(table))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, metric in enumerate(metrics):
        bars = ax.bar(x + (offset - 1) * width, table[metric], width, label=metric)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison at K = {k} (time-based test split)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cold_start(table: pd.DataFrame, k: int, path: Path) -> None:
    """NDCG@k as a function of how many ratings a new user has given."""
    metric = f"NDCG@{k}"
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, group in table.groupby("Strategy", sort=False):
        ax.plot(group["Ratings revealed"], group[metric], marker="o", label=strategy)
    ax.set_xlabel("Number of ratings known for the new user")
    ax.set_ylabel(metric)
    ax.set_title("Cold-start: ranking quality vs. amount of user history")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
