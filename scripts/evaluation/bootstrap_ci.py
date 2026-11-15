#!/usr/bin/env python
"""Paired bootstrap CIs for fixed-split PANDA ensemble comparisons.

Computes ensemble probabilities for each setting (no-gate, mask-only, full-v2)
from member checkpoints, then estimates paired bootstrap confidence intervals
for QWK, accuracy, macro F1, and weighted F1, plus pairwise deltas.

Example (each --*-dirs entry is a run directory written by scripts/training/train_mil.py):
  uv run python scripts/evaluation/bootstrap_ci.py     --nogate-dirs RUN_NOGATE_SEED0 ... RUN_NOGATE_SEED4     --mask-dirs   RUN_MASK_SEED0   ... RUN_MASK_SEED4     --full-dirs   RUN_FULL_SEED0   ... RUN_FULL_SEED4     --embeddings-dir outputs/embeddings_virchow2     --labels-csv data/panda_labels.csv     --split-csv splits/panda_fixed_split.csv     --n-bootstrap 10000     --seed 123     --out-dir outputs/statistical_tests
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EVAL_ROOT = PROJECT_ROOT / "scripts" / "evaluation"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from prostate_mil.training.mil.dataset import SlideEmbeddingDataset
from prostate_mil.training.mil.splits import fixed_split_indices
from ensemble_eval import build_model_from_config


METRICS = ("qwk", "accuracy", "macro_f1", "weighted_f1")
PAIRS = (("full", "mask"), ("full", "nogate"), ("mask", "nogate"))


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def infer_member_probs(
    exp_dir: Path,
    test_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    config_path = exp_dir / "config.json"
    ckpt_path = exp_dir / "best_model.pt"
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config or checkpoint in {exp_dir}")

    config = json.loads(config_path.read_text())
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(config, ckpt["model_state_dict"], device)
    ordinal = config.get("loss_type", "focal") == "coral"

    probs = []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch["embeddings"].squeeze(0).to(device)
            variances = batch["spatial_variances"].squeeze(0).to(device)
            out = model(embeddings, variances)
            if ordinal:
                cum = torch.sigmoid(out["logits"])
                cum = torch.cat(
                    [torch.ones(1, device=cum.device), cum, torch.zeros(1, device=cum.device)]
                )
                p = (cum[:-1] - cum[1:]).cpu().numpy()
                p = np.clip(p, 0.0, None)
                p = p / p.sum()
            else:
                p = F.softmax(out["logits"], dim=-1).cpu().numpy()
            probs.append(p)
    return np.stack(probs, axis=0)


def infer_ensemble_probs(
    exp_dirs: list[Path],
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    member_probs = []
    member_qwks = []
    labels = []

    for i, exp_dir in enumerate(exp_dirs, start=1):
        probs = infer_member_probs(exp_dir, test_loader, device)
        if not labels:
            for batch in test_loader:
                labels.append(int(batch["label"].item()))
        preds = probs.argmax(axis=1)
        qwk = float(cohen_kappa_score(labels, preds, weights="quadratic"))
        member_qwks.append(qwk)
        member_probs.append(probs)
        print(f"  [{i}/{len(exp_dirs)}] {exp_dir.name}: member QWK = {qwk:.4f}")

    stacked = np.stack(member_probs, axis=0)  # [M, N, C]
    ens_probs = stacked.mean(axis=0)
    return ens_probs, member_qwks


def ci_from_samples(values: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.percentile(values, 100.0 * alpha / 2.0))
    hi = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nogate-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--mask-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--full-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--embeddings-dir", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--split-csv", type=Path, default=None,
                    help="Fixed split file (slide_id,split). Overrides the seed-42 rebuild.")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/statistical_tests"))
    args = ap.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed split (CSV, or rebuilt with seed 42)
    dataset = SlideEmbeddingDataset(
        embeddings_dir=args.embeddings_dir,
        labels_csv=args.labels_csv,
    )
    _, _, test_idx = fixed_split_indices(
        dataset, args.val_ratio, args.test_ratio, split_csv=args.split_csv,
    )
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)
    y_true = np.array([int(dataset[i]["label"]) for i in test_idx], dtype=np.int64)

    print(f"Test set size: {len(y_true)}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Device: {device}")

    print("\nComputing no-gate ensemble probabilities...")
    probs_nogate, qwk_nogate_members = infer_ensemble_probs(args.nogate_dirs, test_loader, device)
    print("\nComputing mask-only ensemble probabilities...")
    probs_mask, qwk_mask_members = infer_ensemble_probs(args.mask_dirs, test_loader, device)
    print("\nComputing full-gate ensemble probabilities...")
    probs_full, qwk_full_members = infer_ensemble_probs(args.full_dirs, test_loader, device)

    probs_by_setting = {
        "nogate": probs_nogate,
        "mask": probs_mask,
        "full": probs_full,
    }
    preds_by_setting = {k: v.argmax(axis=1) for k, v in probs_by_setting.items()}

    # Point estimates
    point = {k: metrics_from_preds(y_true, preds) for k, preds in preds_by_setting.items()}

    # Paired bootstrap over slide indices
    rng = np.random.default_rng(args.seed)
    n_samples = len(y_true)
    boot = {
        setting: {m: np.empty(args.n_bootstrap, dtype=np.float64) for m in METRICS}
        for setting in probs_by_setting
    }
    delta = {
        f"{a}_minus_{b}": {m: np.empty(args.n_bootstrap, dtype=np.float64) for m in METRICS}
        for a, b in PAIRS
    }

    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        yb = y_true[idx]
        mb = {}
        for setting, preds in preds_by_setting.items():
            pm = metrics_from_preds(yb, preds[idx])
            mb[setting] = pm
            for m in METRICS:
                boot[setting][m][b] = pm[m]

        for a, c in PAIRS:
            key = f"{a}_minus_{c}"
            for m in METRICS:
                delta[key][m][b] = mb[a][m] - mb[c][m]

    # Summaries with CIs
    ci = {}
    for setting in probs_by_setting:
        ci[setting] = {}
        for m in METRICS:
            lo, hi = ci_from_samples(boot[setting][m], args.alpha)
            ci[setting][m] = {"point": point[setting][m], "ci_low": lo, "ci_high": hi}

    delta_ci = {}
    for key in delta:
        delta_ci[key] = {}
        for m in METRICS:
            lo, hi = ci_from_samples(delta[key][m], args.alpha)
            delta_ci[key][m] = {
                "point": float(np.mean(delta[key][m])),
                "ci_low": lo,
                "ci_high": hi,
            }

    result = {
        "settings": ci,
        "deltas": delta_ci,
        "metadata": {
            "n_test": int(n_samples),
            "n_bootstrap": int(args.n_bootstrap),
            "alpha": float(args.alpha),
            "split_seed": 42,
            "member_qwk": {
                "nogate": qwk_nogate_members,
                "mask": qwk_mask_members,
                "full": qwk_full_members,
            },
            "setting_dirs": {
                "nogate": [str(p) for p in args.nogate_dirs],
                "mask": [str(p) for p in args.mask_dirs],
                "full": [str(p) for p in args.full_dirs],
            },
        },
    }

    out_json = args.out_dir / "panda_fixed_bootstrap.json"
    out_json.write_text(json.dumps(result, indent=2))

    out_csv = args.out_dir / "gate_delta_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "metric", "point", "ci_low", "ci_high"])
        for setting in ci:
            for m in METRICS:
                row = ci[setting][m]
                writer.writerow(["setting", setting, m, row["point"], row["ci_low"], row["ci_high"]])
        for name in delta_ci:
            for m in METRICS:
                row = delta_ci[name][m]
                writer.writerow(["delta", name, m, row["point"], row["ci_low"], row["ci_high"]])

    print("\nSaved:")
    print(f"  {out_json}")
    print(f"  {out_csv}")
    print("\nKey deltas (95% CI):")
    for name in ("full_minus_mask", "full_minus_nogate", "mask_minus_nogate"):
        q = delta_ci[name]["qwk"]
        print(f"  {name} QWK: {q['point']:.4f} [{q['ci_low']:.4f}, {q['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
