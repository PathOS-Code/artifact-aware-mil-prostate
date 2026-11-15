"""Remove optimizer state from best_model.pt files for release.

Training saves the AdamW state with each best checkpoint, which roughly triples
the file size. Inference and the evaluation scripts only read
``model_state_dict`` (plus the run's config.json), so the released weights keep
``model_state_dict``, ``epoch``, ``best_qwk`` and ``config``.

Usage:
  uv run python scripts/training/strip_checkpoint.py RUN_DIR [RUN_DIR ...] --out-root release/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

KEEP_KEYS = ("model_state_dict", "epoch", "best_qwk", "config")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Run directories containing best_model.pt and config.json.")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Each run is written to OUT_ROOT/<run dir name>/.")
    args = ap.parse_args()

    for run_dir in args.run_dirs:
        ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
        slim = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}

        out_dir = args.out_root / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(slim, out_dir / "best_model.pt")
        shutil.copy2(run_dir / "config.json", out_dir / "config.json")

        before = (run_dir / "best_model.pt").stat().st_size / 1e6
        after = (out_dir / "best_model.pt").stat().st_size / 1e6
        print(f"{run_dir.name}: {before:.0f} MB -> {after:.0f} MB")


if __name__ == "__main__":
    main()
