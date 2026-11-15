"""Write the seed-42 train/val/test split as a slide_id,split CSV.

The split is built exactly as in train.py: slides from --labels-csv that have
an embedding file in --embeddings-dir, permuted with torch seed 42, then cut
75/15/10. Val and test rows keep the permutation order.

The published split for the paper is already in splits/panda_fixed_split.csv;
use this script only to regenerate a split for a different embedding set.

Usage:
  uv run python scripts/preprocessing/export_split.py \\
    --embeddings-dir outputs/embeddings_virchow2 \\
    --labels-csv data/panda_labels.csv \\
    --out splits/my_split.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from prostate_mil.training.mil.dataset import SlideEmbeddingDataset
from prostate_mil.training.mil.splits import seeded_split_indices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings-dir", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dataset = SlideEmbeddingDataset(
        embeddings_dir=args.embeddings_dir, labels_csv=args.labels_csv,
    )
    slide_ids = [pt_path.stem for pt_path, _ in dataset.samples]
    train_idx, val_idx, test_idx = seeded_split_indices(
        len(slide_ids), args.val_ratio, args.test_ratio,
    )

    rows = [(slide_ids[i], "train") for i in sorted(train_idx)]
    rows += [(slide_ids[i], "val") for i in val_idx]
    rows += [(slide_ids[i], "test") for i in test_idx]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["slide_id", "split"]).to_csv(
        args.out, index=False, lineterminator="\n",
    )
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} -> {args.out}")


if __name__ == "__main__":
    main()
