"""Paired bootstrap confidence intervals for external ensemble comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap external result deltas.")
    parser.add_argument("--full-json", type=Path, required=True)
    parser.add_argument("--mask-json", type=Path, required=True)
    parser.add_argument("--nogate-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, int]]:
    data = json.loads(path.read_text())
    rows = data.get("slide_predictions")
    if rows is None:
        raise ValueError(f"{path} does not contain slide_predictions. Re-run ensemble_eval.py.")
    return {
        str(row["slide_id"]): {
            "label": int(row["label"]),
            "pred": int(row["ensemble_pred"]),
        }
        for row in rows
    }


def qwk(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(cohen_kappa_score(labels, preds, weights="quadratic"))


def delta_ci(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> dict:
    n = len(labels)
    observed = qwk(labels, preds_a) - qwk(labels, preds_b)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = qwk(labels[idx], preds_a[idx]) - qwk(labels[idx], preds_b[idx])
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_qwk": float(observed),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "crosses_zero": bool(ci_low <= 0.0 <= ci_high),
    }


def main() -> None:
    args = parse_args()
    full = load_predictions(args.full_json)
    mask = load_predictions(args.mask_json)
    nogate = load_predictions(args.nogate_json)

    common = sorted(set(full).intersection(mask).intersection(nogate))
    if not common:
        raise ValueError("No common slide IDs across the three result files.")

    for name, preds in [("full", full), ("mask", mask), ("no_gate", nogate)]:
        missing = sorted(set(common).difference(preds))
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} common slides.")

    labels = np.array([full[sid]["label"] for sid in common], dtype=np.int64)
    for sid in common:
        if mask[sid]["label"] != full[sid]["label"] or nogate[sid]["label"] != full[sid]["label"]:
            raise ValueError(f"Label mismatch for slide {sid}")

    preds_full = np.array([full[sid]["pred"] for sid in common], dtype=np.int64)
    preds_mask = np.array([mask[sid]["pred"] for sid in common], dtype=np.int64)
    preds_nogate = np.array([nogate[sid]["pred"] for sid in common], dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    comparisons = {
        "full_minus_mask": delta_ci(labels, preds_full, preds_mask, args.n_resamples, rng),
        "full_minus_no_gate": delta_ci(labels, preds_full, preds_nogate, args.n_resamples, rng),
        "mask_minus_no_gate": delta_ci(labels, preds_mask, preds_nogate, args.n_resamples, rng),
    }

    result = {
        "n_slides": int(len(common)),
        "n_resamples": int(args.n_resamples),
        "seed": int(args.seed),
        "inputs": {
            "full": str(args.full_json),
            "mask": str(args.mask_json),
            "no_gate": str(args.nogate_json),
        },
        "base_qwk": {
            "full": qwk(labels, preds_full),
            "mask": qwk(labels, preds_mask),
            "no_gate": qwk(labels, preds_nogate),
        },
        "comparisons": comparisons,
        "slide_ids": common,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))

    lines = [
        "CrowdGleason paired bootstrap QWK deltas",
        f"n_slides: {len(common)}",
        f"n_resamples: {args.n_resamples}",
        "",
        "comparison\tdelta_qwk\tci95_low\tci95_high\tcrosses_zero",
    ]
    for name, values in comparisons.items():
        lines.append(
            f"{name}\t{values['delta_qwk']:.6f}\t{values['ci95_low']:.6f}\t"
            f"{values['ci95_high']:.6f}\t{values['crosses_zero']}"
        )
    args.out_json.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print(f"Saved: {args.out_json}")
    print(f"Saved: {args.out_json.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
