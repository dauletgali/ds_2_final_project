"""
visualization.py
----------------
Plotting utilities for Market-1501 re-ID experiments.

Functions
---------
plot_dataset_eda          — images/identity and cameras/identity distributions
plot_similarity_distribution — genuine vs impostor histogram
plot_cmc_curves           — overlay multiple experiments on one CMC chart
plot_reid_results         — top-K retrieval grid for a single query
plot_threshold_sweep      — Rank-1 / mAP vs. cosine threshold
plot_results_comparison   — bar chart comparing two experiments
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Dataset EDA
# ---------------------------------------------------------------------------

def plot_dataset_eda(df: pd.DataFrame, title_suffix: str = "Training Set") -> plt.Figure:
    """
    Reproduce the EDA plots from the original notebook (images per person,
    images per camera, cameras per identity).
    """
    images_per_person = df.groupby("person_id").size()
    images_per_camera = df.groupby("camera_id").size()
    cams_per_person   = df.groupby("person_id")["camera_id"].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Dataset EDA — {title_suffix}", fontsize=12)

    axes[0].hist(images_per_person, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Images per Person")
    axes[0].set_xlabel("# Images")
    axes[0].set_ylabel("Frequency")

    images_per_camera.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="white")
    axes[1].set_title("Images by Camera")
    axes[1].set_xlabel("Camera ID")
    axes[1].set_ylabel("# Images")
    axes[1].tick_params(axis="x", rotation=0)

    cams_per_person.hist(bins=range(1, cams_per_person.max() + 2), ax=axes[2],
                         color="steelblue", edgecolor="white", align="left")
    axes[2].set_title("Cameras per Identity")
    axes[2].set_xlabel("# Cameras")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Similarity distribution
# ---------------------------------------------------------------------------

def plot_similarity_distribution(
    positive_sims: list[float],
    negative_sims: list[float],
    title: str = "Genuine vs Impostor Similarity",
    threshold: float | None = None,
    bins: int = 60,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Histogram of positive (same-ID) vs negative (different-ID) cosine similarities.
    Optionally draws a vertical line at the threshold value.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(negative_sims, bins=bins, alpha=0.55, color="tomato",  label=f"Impostor  (n={len(negative_sims)})")
    ax.hist(positive_sims, bins=bins, alpha=0.65, color="steelblue", label=f"Genuine  (n={len(positive_sims)})")

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                   label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    # Annotate means
    ax.axvline(np.mean(positive_sims), color="steelblue", linestyle=":", linewidth=1.2)
    ax.axvline(np.mean(negative_sims), color="tomato",    linestyle=":", linewidth=1.2)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CMC curves
# ---------------------------------------------------------------------------

def plot_cmc_curves(
    results: dict[str, dict],
    max_rank: int = 10,
    title: str = "CMC Curves",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Overlay multiple experiment CMC curves on one axes.

    Parameters
    ----------
    results : {label: metrics_dict}
        Each metrics_dict must have 'cmc' (np.ndarray) and 'mAP' (float).
    """
    fig, ax = plt.subplots(figsize=figsize)
    ranks = np.arange(1, max_rank + 1)

    for label, res in results.items():
        cmc = res["cmc"][:max_rank]
        ax.plot(ranks, cmc, marker="o", markersize=4,
                label=f"{label}  (mAP={res['mAP']:.3f})")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative Matching Rate")
    ax.set_title(title)
    ax.set_xticks(ranks)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Top-K retrieval grid
# ---------------------------------------------------------------------------

def plot_reid_results(
    query_img_path: str,
    gallery_df: pd.DataFrame,
    gallery_ids: np.ndarray,
    query_pid: int,
    similarities: np.ndarray,
    top_k: int = 5,
    title: str = "Re-ID Top-K Retrieval",
) -> plt.Figure:
    """
    Show a query image and its top-K gallery retrievals.

    Correct matches framed in green, incorrect in red.
    """
    from PIL import Image as PILImage

    sorted_idx = np.argsort(-similarities)
    # Filter out -inf (masked) entries
    sorted_idx = [i for i in sorted_idx if not np.isneginf(similarities[i])][:top_k]

    n_cols  = len(sorted_idx) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Query
    q_img = PILImage.open(query_img_path).convert("RGB")
    axes[0].imshow(q_img)
    axes[0].set_title(f"Query\nID:{query_pid}", fontsize=9)
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("gold")
        spine.set_linewidth(3)
    axes[0].set_visible(True)

    # Gallery matches
    for rank, g_idx in enumerate(sorted_idx, start=1):
        g_path = gallery_df.iloc[g_idx]["image_path"]
        g_id   = gallery_ids[g_idx]
        g_img  = PILImage.open(g_path).convert("RGB")
        ax = axes[rank]
        ax.imshow(g_img)

        is_correct = (g_id == query_pid)
        color = "limegreen" if is_correct else "tomato"
        mark  = "✓" if is_correct else "✗"

        ax.set_title(
            f"Rank {rank} {mark}\nSim:{similarities[g_idx]:.3f}  ID:{g_id}",
            fontsize=8,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def plot_threshold_sweep(
    thresholds: np.ndarray,
    rank1_scores: np.ndarray,
    map_scores: np.ndarray,
    optimal_threshold: float | None = None,
    title: str = "Rank-1 & mAP vs. Cosine Threshold",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Plot how Rank-1 and mAP change across a range of similarity thresholds.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(thresholds, rank1_scores, marker=".", label="Rank-1", color="steelblue")
    ax.plot(thresholds, map_scores,   marker=".", label="mAP",    color="tomato")

    if optimal_threshold is not None:
        ax.axvline(optimal_threshold, color="black", linestyle="--",
                   linewidth=1.2, label=f"Best threshold = {optimal_threshold:.2f}")

    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Results comparison bar chart
# ---------------------------------------------------------------------------

def plot_results_comparison(
    results: dict[str, dict],
    metrics: list[str] = ("rank1", "rank5", "mAP"),
    title: str = "Experiment Comparison",
    figsize: tuple = (9, 5),
) -> plt.Figure:
    """
    Grouped bar chart comparing multiple experiments across metrics.
    Reproduces the results_comparison output from the original notebook.
    """
    labels = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["steelblue", "tomato", "seagreen", "darkorchid"]

    for i, label in enumerate(labels):
        vals = [results[label].get(m, 0) for m in metrics]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=label, color=colors[i % len(colors)])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig
