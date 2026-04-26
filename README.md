# Person Re-Identification — Market-1501

## Project structure

```
reid_project/
├── src/
│   ├── __init__.py
│   ├── dataset.py            ← Market-1501 filename parsing, MarketDataset, load_splits
│   ├── feature_extractor.py  ← build_torchreid_resnet50, extract_embeddings
│   ├── metrics.py            ← cosine_similarity_matrix, apply_threshold, evaluate_ranking
│   └── visualization.py      ← all plots (EDA, CMC, similarity dist, threshold sweep)
├── notebooks/
│   └── experiments.ipynb     ← ALL experiments; imports only from src/
├── outputs/                  ← saved embeddings (.npy) + figures (.png)
├── requirements.txt
└── README.md
```

## Quickstart (Google Colab)

```python
# 1. Mount Drive and set DATASET_ROOT in the Config cell
# 2. Install torchreid
!pip install torchreid

# 3. Open notebooks/experiments.ipynb and run cells top to bottom
```

## Experiments

| # | Name | Key result |
|---|---|---|
| 01 | Cosine similarity baseline | Rank-1 ≈ 10% (ImageNet ResNet50, no fine-tune) |
| 02 | Cosine + feasibility threshold | Varies by threshold |
| Sweep | Threshold sweep | Find optimal operating point |

## Metrics

- **Rank-k**: fraction of queries where the correct identity appears in the top-k results.
- **mAP**: mean Average Precision across all query images (standard re-ID metric).
- Both computed under the **Market-1501 protocol**: same-camera, same-identity pairs are excluded from the gallery during evaluation.

## What's different from the original notebook

| Original | Refactored |
|---|---|
| Filename parsing duplicated in 2 cells (regex + split) | Single `parse_filename()` in `dataset.py` |
| `cosine_similarity()` from sklearn called per-pair in a loop | Vectorised `query_emb @ gallery_emb.T` |
| L2 normalisation done after extraction in a separate cell | Done inside `extract_embeddings()`, verified immediately |
| Evaluation loop written inline 3× (baseline, logistic, context) | Single `evaluate_ranking()` function |
| Embeddings saved/loaded with inconsistent paths | One `OUTPUT_DIR` config variable |
| `to_flat_unit_vector()` defined inline to fix shape bugs | Shape handled correctly inside extractor |
| `baseline_model` (LogisticRegression on cosine_sim) adds no value over raw ranking | Removed; raw cosine is the true baseline |
