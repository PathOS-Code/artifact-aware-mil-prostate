"""Tests for the fixed train/val/test split helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prostate_mil.training.mil.splits import csv_split_indices, seeded_split_indices


def _write_split(path: Path, slide_ids: list[str], parts) -> Path:
    rows = []
    for name, idx in zip(("train", "val", "test"), parts):
        rows += [(slide_ids[i], name) for i in idx]
    pd.DataFrame(rows, columns=["slide_id", "split"]).to_csv(path, index=False)
    return path


def test_seeded_split_sizes_and_disjoint() -> None:
    train, val, test = seeded_split_indices(10614)
    assert (len(train), len(val), len(test)) == (7961, 1592, 1061)
    assert len(set(train) | set(val) | set(test)) == 10614


def test_csv_roundtrip_keeps_order(tmp_path: Path) -> None:
    slide_ids = [f"s{i:03d}" for i in range(100)]
    parts = seeded_split_indices(len(slide_ids))
    csv = _write_split(tmp_path / "split.csv", slide_ids, parts)
    assert csv_split_indices(slide_ids, csv) == parts


def test_missing_train_slide_is_skipped(tmp_path: Path) -> None:
    slide_ids = [f"s{i:03d}" for i in range(100)]
    train, val, test = seeded_split_indices(len(slide_ids))
    csv = _write_split(tmp_path / "split.csv", slide_ids, (train, val, test))

    dropped = slide_ids[train[0]]
    remaining = [s for s in slide_ids if s != dropped]
    new_train, new_val, new_test = csv_split_indices(remaining, csv)

    assert len(new_train) == len(train) - 1
    assert [remaining[i] for i in new_test] == [slide_ids[i] for i in test]
    assert [remaining[i] for i in new_val] == [slide_ids[i] for i in val]


def test_missing_test_slide_raises(tmp_path: Path) -> None:
    slide_ids = [f"s{i:03d}" for i in range(100)]
    parts = seeded_split_indices(len(slide_ids))
    csv = _write_split(tmp_path / "split.csv", slide_ids, parts)

    dropped = slide_ids[parts[2][0]]
    with pytest.raises(FileNotFoundError):
        csv_split_indices([s for s in slide_ids if s != dropped], csv)


def test_published_split_matches_paper_counts() -> None:
    df = pd.read_csv(PROJECT_ROOT / "splits" / "panda_fixed_split.csv")
    counts = df["split"].value_counts().to_dict()
    assert counts == {"train": 7963, "val": 1592, "test": 1061}
    assert df["slide_id"].is_unique
