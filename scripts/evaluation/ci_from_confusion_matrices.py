#!/usr/bin/env python
"""Uncertainty intervals computed from saved ensemble confusion matrices.

No training or inference: reads ``confusion_matrix`` (rows = true grade group
0-5, columns = predicted) from the ensemble result JSONs under results/.

Outputs (JSON):
  (i)  Percentile bootstrap 95% CI of the ensemble QWK: 10,000 resamples of the
       (true, pred) rows (slides / biopsies) with replacement, seed 20261022.
       The recomputed point QWK is checked against the stored ensemble QWK.
  (ii) Clopper-Pearson exact 95% CIs for sensitivity and specificity at
       GG >= 2 (positive = GG 2-5) for the full robust gate ensembles.

Rows are treated as independent (patient clustering is ignored) and the
trained models are held fixed.

Usage:
  uv run python scripts/evaluation/ci_from_confusion_matrices.py \\
    --results-dir results --out outputs/statistical_tests/ci_from_confusion_matrices.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SEED = 20261022
N_BOOT = 10000
K = 6

# Evaluation order matters: all bootstraps draw from one generator in this order.
SOURCES = {
    "panda_full_5seed_focal": "panda/virchow2_transmil_focal_fullgate_5seed.json",
    "sicapv2_nogate_5seed_focal": "sicapv2/virchow2_transmil_focal_nogate_5seed.json",
    "sicapv2_mask_5seed_focal": "sicapv2/virchow2_transmil_focal_maskgate_5seed.json",
    "sicapv2_full_5seed_focal": "sicapv2/virchow2_transmil_focal_fullgate_5seed.json",
    "sicapv2_mask_5fold_coral": "sicapv2/virchow2_transmil_coral_maskgate_5fold.json",
    "sicapv2_full_5fold_coral": "sicapv2/virchow2_transmil_coral_fullgate_5fold.json",
    "crowdgleason_nogate_5seed_focal": "crowdgleason/virchow2_transmil_focal_nogate_5seed.json",
    "crowdgleason_mask_5seed_focal": "crowdgleason/virchow2_transmil_focal_maskgate_5seed.json",
    "crowdgleason_full_5seed_focal": "crowdgleason/virchow2_transmil_focal_fullgate_5seed.json",
    "crowdgleason_mask_5fold_coral": "crowdgleason/virchow2_transmil_coral_maskgate_5fold.json",
    "crowdgleason_full_5fold_coral": "crowdgleason/virchow2_transmil_coral_fullgate_5fold.json",
}
OPERATING_POINT_KEYS = (
    "panda_full_5seed_focal", "sicapv2_full_5seed_focal", "sicapv2_full_5fold_coral",
    "crowdgleason_full_5seed_focal", "crowdgleason_full_5fold_coral",
)


def qwk_from_counts(c) -> float:
    """QWK with quadratic weights from a [K, K] count matrix."""
    c = np.asarray(c, dtype=np.float64)
    n = c.sum()
    idx = np.arange(K)
    w = (idx[:, None] - idx[None, :]) ** 2 / (K - 1) ** 2
    expected = c.sum(axis=1)[:, None] * c.sum(axis=0)[None, :] / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - (w * c).sum() / (w * expected).sum()


def bootstrap_qwk(cm, rng) -> tuple[int, float, float, int]:
    cm = np.array(cm)
    cells = np.repeat(np.arange(K * K), cm.ravel())  # one entry per slide: its (true, pred) cell
    n = len(cells)
    reps = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        reps[b] = qwk_from_counts(np.bincount(cells[idx], minlength=K * K).reshape(K, K))
    reps = reps[~np.isnan(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return n, float(lo), float(hi), int(len(reps))


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Bin(n, p), exact, log-space summation."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    s = 0.0
    for i in range(k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, s)


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided CI by bisection on the binomial CDF."""
    def bisect(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if x == 0 else bisect(lambda p: alpha / 2 - (1.0 - _binom_cdf(x - 1, n, p)))
    upper = 1.0 if x == n else bisect(lambda p: _binom_cdf(x, n, p) - alpha / 2)
    return lower, upper


def operating_point(cm) -> tuple[int, int, int, int]:
    """(true positives, positives, true negatives, negatives) at GG >= 2."""
    cm = np.array(cm)
    return int(cm[2:, 2:].sum()), int(cm[2:, :].sum()), int(cm[:2, :2].sum()), int(cm[:2, :].sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("outputs/statistical_tests/ci_from_confusion_matrices.json"))
    args = ap.parse_args()

    stored = {key: json.loads((args.results_dir / rel).read_text()) for key, rel in SOURCES.items()}
    cms = {key: d["confusion_matrix"] for key, d in stored.items()}

    rng = np.random.default_rng(SEED)
    out = {
        "meta": {
            "bootstrap": "percentile 95% CI, resampling (true, pred) rows with replacement",
            "n_resamples": N_BOOT, "seed": SEED, "numpy": np.__version__,
            "clopper_pearson": "exact two-sided 95%",
            "caveat": "rows treated as independent; patient clustering ignored; trained models held fixed",
            "sources": SOURCES,
        },
        "qwk": {}, "operating_points": {},
    }
    for key, cm in cms.items():
        point = float(qwk_from_counts(cm))
        reported = float(stored[key]["ensemble_test_qwk"])
        assert abs(point - reported) < 5e-4, (key, point, reported)
        n, lo, hi, n_valid = bootstrap_qwk(cm, rng)
        out["qwk"][key] = {"n": n, "qwk_recomputed": point, "qwk_reported": reported,
                           "ci_low": lo, "ci_high": hi, "valid_resamples": n_valid}

    for key in OPERATING_POINT_KEYS:
        tp, n_pos, tn, n_neg = operating_point(cms[key])
        entry = {}
        for name, x, n in (("sensitivity", tp, n_pos), ("specificity", tn, n_neg)):
            lo, hi = clopper_pearson(x, n)
            entry[name] = {"x": x, "n": n, "estimate": x / n, "ci_low": lo, "ci_high": hi}
        out["operating_points"][key] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("wrote", args.out)
    for k, v in out["qwk"].items():
        print(f"{k:34s} n={v['n']:5d} QWK {v['qwk_recomputed']:.4f} CI [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    for k, v in out["operating_points"].items():
        for m, e in v.items():
            print(f"{k:34s} {m:11s} {e['x']}/{e['n']} = {e['estimate']:.4f} CI [{e['ci_low']:.4f}, {e['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
