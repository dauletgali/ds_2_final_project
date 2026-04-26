from .dataset import build_dataframe, load_splits, MarketDataset, print_split_stats, REID_TRANSFORM
from .feature_extractor import (
    build_torchvision_resnet50,
    build_torchreid_resnet50,
    extract_embeddings,
    verify_normalization,
)
from .metrics import (
    cosine_similarity_matrix,
    build_invalid_mask,
    apply_threshold,
    evaluate_ranking,
    print_metrics,
    sample_pair_similarities,
)
from .visualization import (
    plot_dataset_eda,
    plot_similarity_distribution,
    plot_cmc_curves,
    plot_reid_results,
    plot_threshold_sweep,
    plot_results_comparison,
)