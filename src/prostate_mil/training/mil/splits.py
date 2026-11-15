"""Fixed PANDA train/val/test split shared by training and evaluation scripts.

Two ways to build the split:

- From ``splits/panda_fixed_split.csv`` (``slide_id,split``). This is the
  published split used in the paper. Rows are stored in the original
  permutation order, so evaluation (including bootstrap resampling) iterates
  the test slides in the same order as the original runs.
- From ``torch.randperm`` with seed 42 over the slides that have an embedding
  file. This is how the split was first produced. It only reproduces the
  published split when exactly the same 10,614 slides have embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

logger = logging.getLogger(__name__)

SPLIT_SEED = 42
SPLIT_NAMES = ("train", "val", "test")


def seeded_split_indices(
    n_total: int, val_ratio: float = 0.15, test_ratio: float = 0.10,
) -> tuple[list[int], list[int], list[int]]:
    """Seed-42 permutation split over ``n_total`` slides."""
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    indices = torch.randperm(n_total, generator=generator).tolist()
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:],
    )


def csv_split_indices(
    slide_ids: list[str], split_csv: str | Path,
) -> tuple[list[int], list[int], list[int]]:
    """Map a ``slide_id,split`` CSV onto dataset indices, keeping CSV order.

    Training slides without an embedding are skipped with a warning.
    Validation or test slides without an embedding raise an error, because
    the reported metrics would no longer be comparable.
    """
    df = pd.read_csv(split_csv, dtype={"slide_id": str})
    unknown = set(df["split"]) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"Unknown split names in {split_csv}: {sorted(unknown)}")

    position = {slide_id: i for i, slide_id in enumerate(slide_ids)}
    out: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    missing: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for slide_id, split in zip(df["slide_id"], df["split"]):
        idx = position.get(slide_id)
        if idx is None:
            missing[split].append(slide_id)
        else:
            out[split].append(idx)

    for split in ("val", "test"):
        if missing[split]:
            raise FileNotFoundError(
                f"{len(missing[split])} {split} slides from {split_csv} have no "
                f"embedding, e.g. {missing[split][:3]}"
            )
    if missing["train"]:
        logger.warning(
            "%d training slides from %s have no embedding and are skipped: %s",
            len(missing["train"]), split_csv, missing["train"],
        )

    extra = len(slide_ids) - sum(len(v) for v in out.values())
    if extra:
        logger.warning("%d slides with embeddings are not listed in %s", extra, split_csv)

    return out["train"], out["val"], out["test"]


def fixed_split_indices(
    dataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    split_csv: str | Path | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train, val, test) indices into a ``SlideEmbeddingDataset``."""
    if split_csv is not None:
        slide_ids = [pt_path.stem for pt_path, _ in dataset.samples]
        return csv_split_indices(slide_ids, split_csv)
    return seeded_split_indices(len(dataset), val_ratio, test_ratio)
