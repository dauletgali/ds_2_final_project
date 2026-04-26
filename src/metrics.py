"""
metrics.py
----------
Evaluation metrics for Market-1501 person re-identification.

Implements:
  - cosine_similarity_matrix   : fast dot-product similarity (L2-norm assumed)
  - build_invalid_mask         : Market-1501 same-cam/same-ID exclusion rule
  - apply_threshold            : feasibility threshold filtering
  - evaluate_ranking           : Rank-1, Rank-5, Rank-10, mAP
  - sample_pair_similarities   : genuine/impostor distribution analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity via dot product.

    Assumes both matrices are already L2-normalised (norm ≈ 1.0 per row).
    Using dot product instead of sklearn's cosine_similarity avoids
    re-normalisation and is ~10× faster for large matrices.

    Parameters
    ----------
    query_emb   : np.ndarray, shape (Nq, D)
    gallery_emb : np.ndarray, shape (Ng, D)

    Returns
    -------
    sim_matrix : np.ndarray, shape (Nq, Ng), values in [-1, 1]
    """
    return query_emb @ gallery_emb.T


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def build_invalid_mask(
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: np.ndarray,
    gallery_cams: np.ndarray,
) -> np.ndarray:
    """
    Build the standard Market-1501 invalid-match mask.

    A gallery entry is invalid for a query if it has the SAME person_id
    AND the SAME camera_id (i.e., the image was taken by the same camera
    at the same time — not a genuine cross-camera match).

    Returns
    -------
    mask : np.ndarray bool, shape (Nq, Ng)
        True where the gallery entry should be excluded.
    """
    same_id  = query_ids[:, None] == gallery_ids[None, :]
    same_cam = query_cams[:, None] == gallery_cams[None, :]
    return same_id & same_cam


def apply_threshold(
    sim_matrix: np.ndarray,
    threshold: float,
    fill_value: float = -np.inf,
) -> np.ndarray:
    """
    Mask gallery entries whose similarity is below `threshold`.

    Entries below the threshold are set to `fill_value` so they rank
    last and are never returned as valid matches.

    Parameters
    ----------
    sim_matrix : np.ndarray, shape (Nq, Ng)
    threshold  : float  — minimum cosine similarity to be considered feasible
    fill_value : float  — value assigned to infeasible entries (default -inf)
    """
    out = sim_matrix.copy()
    out[out < threshold] = fill_value
    return out


# ---------------------------------------------------------------------------
# Ranking evaluation
# ---------------------------------------------------------------------------

def evaluate_ranking(
    sim_matrix: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: np.ndarray,
    gallery_cams: np.ndarray,
    max_rank: int = 10,
) -> dict:
    """
    Compute CMC and mAP using the standard Market-1501 evaluation protocol.

    The invalid mask (same cam + same ID) is applied inside this function,
    so you can pass the raw sim_matrix without pre-masking.

    Parameters
    ----------
    sim_matrix   : np.ndarray, shape (Nq, Ng)
    query_ids    : np.ndarray, shape (Nq,)
    gallery_ids  : np.ndarray, shape (Ng,)
    query_cams   : np.ndarray, shape (Nq,)
    gallery_cams : np.ndarray, shape (Ng,)
    max_rank     : int

    Returns
    -------
    dict with keys: cmc, mAP, rank1, rank5, rank10, valid_queries
    """
    invalid_mask = build_invalid_mask(query_ids, gallery_ids, query_cams, gallery_cams)
    masked = sim_matrix.copy()
    masked[invalid_mask] = -np.inf

    num_query = sim_matrix.shape[0]
    cmc_counts = np.zeros(max_rank, dtype=float)
    all_ap = []
    valid_queries = 0

    for i in range(num_query):
        scores = masked[i]

        if np.all(np.isneginf(scores)):
            continue

        sorted_idx    = np.argsort(-scores)
        sorted_ids    = gallery_ids[sorted_idx]
        sorted_scores = scores[sorted_idx]

        true_id  = query_ids[i]
        is_match = (sorted_ids == true_id).astype(int)

        # Remove -inf entries before computing AP (they're masked-out)
        valid_mask = ~np.isneginf(sorted_scores)
        if is_match[valid_mask].sum() == 0:
            continue

        ap = average_precision_score(is_match[valid_mask], sorted_scores[valid_mask])
        all_ap.append(ap)

        # CMC: credit the first rank at which a true match appears
        first_match = np.where(is_match == 1)[0]
        if len(first_match) > 0:
            hit_rank = first_match[0]
            if hit_rank < max_rank:
                cmc_counts[hit_rank:] += 1

        valid_queries += 1

    cmc = cmc_counts / valid_queries if valid_queries > 0 else cmc_counts
    mAP = float(np.mean(all_ap)) if all_ap else 0.0

    return {
        "cmc":           cmc,
        "mAP":           mAP,
        "rank1":         float(cmc[0]),
        "rank5":         float(cmc[min(4, max_rank - 1)]),
        "rank10":        float(cmc[min(9, max_rank - 1)]),
        "valid_queries": valid_queries,
    }


def print_metrics(results: dict, label: str = "Results") -> None:
    """Pretty-print a results dict from evaluate_ranking()."""
    print(f"\n{'=' * 44}")
    print(f"  {label}")
    print(f"{'=' * 44}")
    print(f"  Rank-1  : {results['rank1']:.4f}")
    print(f"  Rank-5  : {results['rank5']:.4f}")
    print(f"  Rank-10 : {results['rank10']:.4f}")
    print(f"  mAP     : {results['mAP']:.4f}")
    print(f"  Queries : {results['valid_queries']}")
    print(f"{'=' * 44}\n")


# ---------------------------------------------------------------------------
# Pair-level similarity analysis (for EDA / distribution plots)
# ---------------------------------------------------------------------------

def sample_pair_similarities(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_pairs: int = 2000,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """
    Sample genuine (same-ID) and impostor (different-ID) cosine similarities.

    Used for distribution analysis and threshold selection — this is the
    analysis from the early part of the original notebook, cleaned up and
    made reproducible.

    Parameters
    ----------
    df         : pd.DataFrame with 'person_id' column
    embeddings : np.ndarray, shape (N, D), L2-normalised
    n_pairs    : int — number of pairs to sample for each class
    seed       : int

    Returns
    -------
    positive_sims, negative_sims : lists of float
    """
    rng = np.random.default_rng(seed)

    person_groups = df.groupby("person_id").indices
    all_indices   = np.arange(len(df))
    person_ids    = df["person_id"].to_numpy()

    positive_sims = []
    negative_sims = []

    unique_pids = list(person_groups.keys())

    # Genuine pairs
    for _ in range(n_pairs):
        pid  = rng.choice(unique_pids)
        idxs = list(person_groups[pid])
        if len(idxs) < 2:
            continue
        i, j = rng.choice(idxs, size=2, replace=False)
        sim = float(embeddings[i] @ embeddings[j])
        positive_sims.append(sim)

    # Impostor pairs
    attempts = 0
    while len(negative_sims) < n_pairs and attempts < n_pairs * 10:
        i, j = rng.choice(all_indices, size=2, replace=False)
        if person_ids[i] != person_ids[j]:
            sim = float(embeddings[i] @ embeddings[j])
            negative_sims.append(sim)
        attempts += 1

    return positive_sims, negative_sims
