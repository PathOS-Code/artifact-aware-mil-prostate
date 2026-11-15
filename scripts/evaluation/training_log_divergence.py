"""First epoch at which logged train/val QWK and loss differ between same-seed
members of the Virchow2 TransMIL-style five-seed sets:
(a) mask-only vs full robust gate, (b) ungated vs each gated set.
No training; reads results/runs/*/seed*/training_log.json and config.json only.

Usage:
  uv run python scripts/evaluation/training_log_divergence.py [RUNS_ROOT] [OUT_JSON]
"""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "runs")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("outputs", "statistical_tests", "training_log_divergence.json")
paths = {
    "nogate": os.path.join(root, "virchow2_transmil_focal_nogate_5seed", "seed{s}"),
    "maskonly": os.path.join(root, "virchow2_transmil_focal_maskgate_5seed", "seed{s}"),
    "full": os.path.join(root, "virchow2_transmil_focal_fullgate_5seed", "seed{s}"),
}

def load(arm, s):
    d = os.path.join(paths[arm].format(s=s))
    log = json.load(open(os.path.join(d, "training_log.json")))
    cfg = json.load(open(os.path.join(d, "config.json")))
    return log, cfg

def first_diff(a, b):
    """1-based first epoch where values differ exactly; None if identical over the common length."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i + 1
    return None

def compare(arm_a, arm_b):
    rows = []
    for s in range(5):
        la, ca = load(arm_a, s)
        lb, cb = load(arm_b, s)
        r = {"seed": s, "init_seed_a": ca.get("init_seed"), "init_seed_b": cb.get("init_seed"),
             "n_epochs_a": len(la["train_qwk"]), "n_epochs_b": len(lb["train_qwk"])}
        for k in ("train_qwk", "val_qwk", "train_loss", "val_loss"):
            r["first_diff_" + k] = first_diff(la[k], lb[k])
        # maximum absolute QWK difference over first 5 epochs for context
        r["max_abs_diff_val_qwk_epochs_1_5"] = max(abs(x - y) for x, y in zip(la["val_qwk"][:5], lb["val_qwk"][:5]))
        rows.append(r)
    return rows

res = {
    "note": "first_diff_* = first 1-based epoch at which the logged values differ exactly (None = identical for all common epochs). Read-only analysis of training_log.json.",
    "a_maskonly_vs_full": compare("maskonly", "full"),
    "b_nogate_vs_maskonly": compare("nogate", "maskonly"),
    "b_nogate_vs_full": compare("nogate", "full"),
}
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
json.dump(res, open(out, "w"), indent=2)
for k in ("a_maskonly_vs_full", "b_nogate_vs_maskonly", "b_nogate_vs_full"):
    print(k)
    for r in res[k]:
        print("  seed", r["seed"], "init", r["init_seed_a"], r["init_seed_b"],
              "train_qwk", r["first_diff_train_qwk"], "val_qwk", r["first_diff_val_qwk"],
              "train_loss", r["first_diff_train_loss"], "val_loss", r["first_diff_val_loss"])
