"""
feature_extractor.py
--------------------
Feature extraction for Market-1501 person re-identification.

Unifies the two model approaches from the original notebook:
  1. Plain torchvision ResNet50 (ImageNet weights, head removed)
  2. torchreid ResNet50 (Market-1501 Re-ID pretrained weights)

Both expose the same extract_embeddings() interface so experiments are
trivially swappable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pandas as pd


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_torchvision_resnet50(device: str = "cuda") -> nn.Module:
    """
    ResNet50 with ImageNet weights, classification head removed.
    Output: 2048-dim feature vector per image.
    """
    from torchvision import models
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(resnet.children())[:-1])  # remove AvgPool + FC
    model = model.to(device)
    model.eval()
    return model


def build_torchreid_resnet50(num_classes: int = 751, device: str = "cuda") -> nn.Module:
    """
    torchreid ResNet50 with Market-1501 pretrained Re-ID weights.
    Classifier head is replaced with Identity so output is the 2048-dim embedding.

    Parameters
    ----------
    num_classes : int
        Must match the dataset used for pretraining (751 for Market-1501).
    """
    try:
        import torchreid
    except ImportError:
        raise ImportError(
            "torchreid is required. Install with:\n"
            "  pip install torchreid\n"
            "or: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
        )

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=num_classes,
        loss="softmax",
        pretrained=True,
    )
    model.classifier = nn.Identity()  # strip head → raw 2048-d embedding
    model = model.to(device)
    model.eval()

    # torch.compile gives free speed on A100 / H100 (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("torch.compile applied — graph optimisation active")
    except Exception:
        pass

    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    df: pd.DataFrame,
    model: nn.Module,
    transform=None,
    batch_size: int = 512,
    device: str = "cuda",
    use_amp: bool = True,
) -> np.ndarray:
    """
    Extract L2-normalised embeddings for every image in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Must have an 'image_path' column.
    model : nn.Module
        Feature extractor; output shape (B, D) or (B, D, 1, 1).
    transform : torchvision transform, optional
        Defaults to the standard 256×128 ImageNet-normalised transform.
    batch_size : int
        Larger values are faster on GPU; 512 is safe for A100 + ResNet50.
    device : str
        'cuda' or 'cpu'.
    use_amp : bool
        Use float16 AMP on CUDA (free ~2× speedup on Tensor Cores).

    Returns
    -------
    np.ndarray of shape (N, D), each row L2-normalised (norm ≈ 1.0).
    """
    from src.dataset import MarketDataset, REID_TRANSFORM

    dataset = MarketDataset(df, transform=transform or REID_TRANSFORM)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    # Probe embedding dimension with a single forward pass
    probe = next(iter(loader))[0][:1].to(device)
    with torch.no_grad():
        out = model(probe)
    emb_dim = out.view(1, -1).shape[1]

    all_embeddings = np.zeros((len(df), emb_dim), dtype=np.float32)

    autocast_ctx = (
        torch.cuda.amp.autocast() if (use_amp and device == "cuda")
        else torch.no_grad()
    )

    with torch.no_grad(), autocast_ctx:
        for images, indices in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device, non_blocking=True)
            feats = model(images)
            feats = feats.view(feats.size(0), -1)   # (B, D)

            # L2 normalise on GPU before moving to CPU
            feats = nn.functional.normalize(feats, p=2, dim=1)
            all_embeddings[indices.numpy()] = feats.cpu().numpy()

    return all_embeddings


def verify_normalization(embeddings: np.ndarray) -> None:
    """Print norm statistics — all values should be ~1.0 after L2 normalisation."""
    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"Embedding norms — Mean: {norms.mean():.4f} | "
        f"Min: {norms.min():.4f} | Max: {norms.max():.4f}"
    )
