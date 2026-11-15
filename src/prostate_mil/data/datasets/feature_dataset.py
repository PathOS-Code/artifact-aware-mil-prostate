"""Dataset PyTorch pour lecture conjointe des patchs et de leur CSV.

Multi-slide capable with LRU WSI cache to prevent memory explosion.
Each worker keeps only 1 WSI open at a time — when it needs a new one,
it closes the previous one first. This bounds memory to ~40MB per worker
instead of accumulating hundreds of memory-mapped files.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from prostate_mil.data.wsi.wsi_reader import WSIReader


class _LRUReaderCache:
    """Least-Recently-Used cache for WSI readers.

    Keeps at most `max_size` WSI files open. When a new file is opened,
    the oldest one is explicitly closed, freeing the memory-mapped handle.
    """

    def __init__(self, max_size: int = 1) -> None:
        self._cache: OrderedDict[int, WSIReader] = OrderedDict()
        self.max_size = max_size

    def get(self, slide_idx: int, wsi_path: Path) -> WSIReader:
        if slide_idx in self._cache:
            self._cache.move_to_end(slide_idx)
            return self._cache[slide_idx]

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            _, old_reader = self._cache.popitem(last=False)
            old_reader.close()  # Explicitly free memory-mapped file!

        # Open new reader
        reader = WSIReader(wsi_path)
        reader.open()
        self._cache[slide_idx] = reader
        return reader


class PatchEmbeddingDataset(Dataset[dict[str, Tensor | float | str]]):
    """Single-slide dataset for backward compatibility."""

    def __init__(
        self,
        csv_path: str | Path,
        wsi_path: str | Path,
        transform: Callable | None = None,
        use_macenko: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.wsi_path = Path(wsi_path)
        self.records = pd.read_csv(self.csv_path)
        self.transform = transform
        self.reader: WSIReader | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | float | str]:
        if self.reader is None:
            self.reader = WSIReader(self.wsi_path)
            self.reader.open()

        row = self.records.iloc[index]
        patch_rgb = self.reader.read_region_rgb(
            location=(int(row["coord_x"]), int(row["coord_y"])),
            level=0, size=(256, 256),
        )

        image = Image.fromarray(patch_rgb)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "patch_id": str(row["patch_id"]),
            "coord_x": float(row["coord_x"]),
            "coord_y": float(row["coord_y"]),
            "variance_spatiale": float(row["variance_spatiale"]),
        }


class MultiSlidePatchDataset(Dataset[dict[str, Tensor | float | str]]):
    """Multi-slide dataset with LRU WSI cache.

    Memory-safe: each worker keeps only 1 WSI open at a time (~40MB).
    Uses flat numpy arrays for the index (~5MB per 100K patches).
    """

    def __init__(
        self,
        slide_entries: list[dict],
        transform: Callable | None = None,
    ) -> None:
        self.transform = transform
        self._skipped: int = 0

        # Store only paths (tiny memory)
        self._wsi_paths: list[Path] = [
            Path(e["wsi_path"]) for e in slide_entries
        ]

        # Read all CSVs and flatten into numpy arrays
        all_slide_idx = []
        all_coord_x = []
        all_coord_y = []
        all_variance = []
        all_patch_id = []

        for slide_idx, entry in enumerate(slide_entries):
            try:
                df = pd.read_csv(entry["csv_path"])
                if len(df) == 0:
                    self._skipped += 1
                    continue
            except Exception:
                self._skipped += 1
                continue

            n = len(df)
            all_slide_idx.extend([slide_idx] * n)
            all_coord_x.extend(df["coord_x"].values.tolist())
            all_coord_y.extend(df["coord_y"].values.tolist())
            all_variance.extend(df["variance_spatiale"].values.tolist())
            all_patch_id.extend(df["patch_id"].astype(str).values.tolist())

        # Compact numpy storage
        self._slide_idx = np.array(all_slide_idx, dtype=np.int32)
        self._coord_x = np.array(all_coord_x, dtype=np.int32)
        self._coord_y = np.array(all_coord_y, dtype=np.int32)
        self._variance = np.array(all_variance, dtype=np.float32)
        self._patch_id = all_patch_id

        del all_slide_idx, all_coord_x, all_coord_y, all_variance, all_patch_id

        # LRU cache is created per-worker (not shared)
        # Initialized lazily in __getitem__ so each spawned worker gets its own
        self._reader_cache: _LRUReaderCache | None = None

    def __len__(self) -> int:
        return len(self._slide_idx)

    def __getitem__(self, index: int) -> dict[str, Tensor | float | str]:
        # Lazy-init LRU cache per worker process
        if self._reader_cache is None:
            self._reader_cache = _LRUReaderCache(max_size=1)

        slide_idx = int(self._slide_idx[index])
        reader = self._reader_cache.get(slide_idx, self._wsi_paths[slide_idx])

        coord_x = int(self._coord_x[index])
        coord_y = int(self._coord_y[index])

        patch_rgb = reader.read_region_rgb(
            location=(coord_x, coord_y), level=0, size=(256, 256),
        )

        image = Image.fromarray(patch_rgb)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "patch_id": self._patch_id[index],
            "coord_x": float(coord_x),
            "coord_y": float(coord_y),
            "variance_spatiale": float(self._variance[index]),
            "slide_idx": slide_idx,
        }
