"""
dataset.py
----------
Market-1501 data loading and parsing.

Consolidates the two different parsing approaches from the original notebook
into one canonical, well-tested implementation.

Market-1501 filename convention:
    <pid>_c<camid>s<seqid>_<frame>_<detid>.jpg
    e.g.  0001_c1s1_000151_01.jpg  →  pid=1, cam=1, seq=1, frame=151
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

_FNAME_PATTERN = re.compile(r"^([-\d]+)_c(\d)s(\d+)_(\d+)_\d+\.jpg$")


def parse_filename(fname: str) -> dict | None:
    """
    Parse a single Market-1501 filename.

    Returns a dict with keys person_id, camera_id, sequence_id, frame,
    or None for junk images (person_id == -1) and non-matching filenames.
    """
    m = _FNAME_PATTERN.match(os.path.basename(fname))
    if not m:
        return None
    pid = int(m.group(1))
    if pid == -1:
        return None
    return {
        "person_id":   pid,
        "camera_id":   int(m.group(2)),
        "sequence_id": int(m.group(3)),
        "frame":       int(m.group(4)),
    }


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_dataframe(directory: str | Path) -> pd.DataFrame:
    """
    Scan a directory of Market-1501 images and return a tidy DataFrame.

    Parameters
    ----------
    directory : str or Path
        One of: bounding_box_train, query, bounding_box_test

    Returns
    -------
    pd.DataFrame with columns:
        image_path, filename, person_id, camera_id, sequence_id, frame
    """
    directory = Path(directory)
    rows = []
    for img_path in sorted(directory.glob("*.jpg")):
        parsed = parse_filename(img_path.name)
        if parsed is None:
            continue
        rows.append({"image_path": str(img_path), "filename": img_path.name, **parsed})

    df = pd.DataFrame(rows)
    return df


def load_splits(dataset_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train / query / gallery splits from the Market-1501 root directory.

    Parameters
    ----------
    dataset_root : path to the Market-1501-v15.09.15 folder

    Returns
    -------
    train_df, query_df, gallery_df
    """
    root = Path(dataset_root)
    train_df   = build_dataframe(root / "bounding_box_train")
    query_df   = build_dataframe(root / "query")
    gallery_df = build_dataframe(root / "bounding_box_test")
    return train_df, query_df, gallery_df


def print_split_stats(df: pd.DataFrame, name: str) -> None:
    """Print a quick summary for a split DataFrame."""
    print(f"\n[{name}]")
    print(f"  Images      : {len(df):,}")
    print(f"  Identities  : {df['person_id'].nunique():,}")
    print(f"  Cameras     : {df['camera_id'].nunique()}")
    cams_pp = df.groupby("person_id")["camera_id"].nunique()
    print(f"  Avg cams/ID : {cams_pp.mean():.2f}")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

REID_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class MarketDataset(Dataset):
    """
    PyTorch Dataset for a Market-1501 split DataFrame.

    Returns (tensor_image, row_index) so that embeddings can be placed
    back in the correct position without shuffle-related bugs.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or REID_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.df.loc[idx, "image_path"]
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx
