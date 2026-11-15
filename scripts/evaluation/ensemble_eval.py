"""Ensemble evaluation: averages softmax probabilities from N checkpoints on the
FIXED test split (splits/panda_fixed_split.csv, or rebuilt with seed 42).

All checkpoints must have been trained on the same split. Only model init seed
varies across ensemble members.

Usage:
  uv run python scripts/evaluation/ensemble_eval.py \\
    --exp-dirs exp1 exp2 exp3 exp4 exp5 \\
    --embeddings-dir outputs/embeddings_virchow2 \\
    --labels-csv data/panda_labels.csv \\
    --split-csv splits/panda_fixed_split.csv \\
    --out outputs/experiments/ensemble_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Subset

from prostate_mil.config.defaults import GatingConfig, TrainingConfig
from prostate_mil.models.mil.model import CustomMILModel
from prostate_mil.training.mil.dataset import SlideEmbeddingDataset
from prostate_mil.training.mil.losses import coral_predict
from prostate_mil.training.mil.splits import fixed_split_indices


def build_model_from_config(config: dict, state_dict: dict, device: torch.device) -> CustomMILModel:
    # Infer input_dim from the saved attention weight (robust to backbone naming drift)
    if "attention.attention_v.weight" in state_dict:
        input_dim = state_dict["attention.attention_v.weight"].shape[1]
    elif "attention.pooling.attention_v.weight" in state_dict:
        input_dim = state_dict["attention.pooling.attention_v.weight"].shape[1]
    else:
        input_dim = GatingConfig().input_dim

    # Resolve aggregator
    transmil_layers = config.get("transmil_layers", 2)
    if transmil_layers == "N/A":
        aggregator = "abmil"
        transmil_layers = 2
    else:
        aggregator = "transmil"

    _mode_aliases = {"v1": "standard", "v2": "robust", "N/A": "robust"}
    gating_mode = config.get("gating_mode", "robust")
    gating_mode = _mode_aliases.get(gating_mode, gating_mode)

    gc = GatingConfig()
    ordinal = config.get("loss_type", "focal") == "coral"
    model = CustomMILModel(
        input_dim=input_dim,
        attention_dim=gc.attention_dim,
        num_classes=TrainingConfig().num_classes,
        variance_threshold=config.get("variance_threshold", gc.variance_threshold),
        alpha=gc.alpha,
        use_gating=config.get("gating", True),
        gating_mode=gating_mode,
        dropout=config.get("dropout", 0.25),
        patch_dropout=0.0,  # inference — no stochastic patch drop
        aggregator=aggregator,
        transmil_layers=transmil_layers,
        ordinal=ordinal,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _batch_slide_name(batch: dict) -> str:
    slide_name = batch.get("slide_name", "")
    if isinstance(slide_name, (list, tuple)):
        return str(slide_name[0])
    return str(slide_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", nargs="+", required=True, type=Path,
                    help="Experiment dirs to ensemble (each needs best_model.pt + config.json).")
    ap.add_argument("--embeddings-dir", type=Path, required=True)
    ap.add_argument("--labels-csv", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--split-csv", type=Path, default=None,
                    help="Fixed split file (slide_id,split). Overrides the seed-42 rebuild.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    dev = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    # Rebuild the EXACT same seed-42 split used during training
    full = SlideEmbeddingDataset(embeddings_dir=args.embeddings_dir,
                                 labels_csv=args.labels_csv)
    n = len(full)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)

    if args.split_csv is None and n_val == 0 and n_test == 0:
        # External dataset mode: evaluate on the entire dataset
        test_loader = DataLoader(full, batch_size=1, shuffle=False)
        print(f"Evaluating on entire dataset ({n} slides)")
    else:
        _, _, test_idx = fixed_split_indices(
            full, args.val_ratio, args.test_ratio, split_csv=args.split_csv,
        )
        test_loader = DataLoader(Subset(full, test_idx), batch_size=1, shuffle=False)
        source = args.split_csv or "seed 42"
        print(f"Test set: {len(test_idx)} slides (from {source})")

    summed_probs = None
    labels: list[int] = []
    slide_names: list[str] = []
    per_member_qwks: list[float] = []
    per_member_predictions: list[dict] = []

    for i, exp_dir in enumerate(args.exp_dirs):
        config_path = exp_dir / "config.json"
        ckpt_path = exp_dir / "best_model.pt"
        if not config_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(f"Missing config or checkpoint in {exp_dir}")

        config = json.loads(config_path.read_text())
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model = build_model_from_config(config, ckpt["model_state_dict"], dev)

        ordinal = config.get("loss_type", "focal") == "coral"
        member_probs = []
        member_labels = []
        member_slide_names = []
        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch["embeddings"].squeeze(0).to(dev)
                variances = batch["spatial_variances"].squeeze(0).to(dev)
                out = model(embeddings, variances)
                if ordinal:
                    # Convert K-1 cumulative sigmoids to K-class probabilities
                    # P(class=k) = σ(k) - σ(k+1), with σ(0)=1 and σ(K)=0
                    cum = torch.sigmoid(out["logits"])  # [K-1]
                    cum = torch.cat([torch.ones(1, device=cum.device), cum,
                                     torch.zeros(1, device=cum.device)])
                    probs = (cum[:-1] - cum[1:]).cpu().numpy()  # [K]
                    probs = np.clip(probs, 0.0, None)  # numerical safety
                    probs = probs / probs.sum()
                else:
                    probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
                member_probs.append(probs)
                member_labels.append(int(batch["label"].item()))
                member_slide_names.append(_batch_slide_name(batch))

        member_probs = np.stack(member_probs, axis=0)  # [N_test, num_classes]
        if summed_probs is None:
            summed_probs = member_probs
            labels = member_labels
            slide_names = member_slide_names
        else:
            if member_slide_names != slide_names:
                raise ValueError(
                    f"Slide order mismatch for {exp_dir}. "
                    "All ensemble members must predict the same slide set in the same order."
                )
            if member_labels != labels:
                raise ValueError(f"Label mismatch for {exp_dir}.")
            summed_probs = summed_probs + member_probs

        member_preds = member_probs.argmax(axis=1)
        q = cohen_kappa_score(member_labels, member_preds, weights="quadratic")
        per_member_qwks.append(float(q))
        per_member_predictions.append(
            {
                "exp_dir": str(exp_dir),
                "qwk": float(q),
                "predictions": [
                    {
                        "slide_id": slide_names[j],
                        "label": int(labels[j]),
                        "pred": int(member_preds[j]),
                    }
                    for j in range(len(labels))
                ],
            }
        )
        print(f"  [{i+1}/{len(args.exp_dirs)}] {exp_dir.name}: single-model test QWK = {q:.4f}")

    avg_probs = summed_probs / len(args.exp_dirs)
    ens_preds = avg_probs.argmax(axis=1)
    ens_qwk = float(cohen_kappa_score(labels, ens_preds, weights="quadratic"))
    ens_acc = float((ens_preds == np.array(labels)).mean())
    ens_macro_f1 = float(f1_score(labels, ens_preds, labels=list(range(6)), average="macro", zero_division=0))
    ens_weighted_f1 = float(
        f1_score(labels, ens_preds, labels=list(range(6)), average="weighted", zero_division=0)
    )

    report_txt = classification_report(labels, ens_preds,
                                       target_names=[f"ISUP {i}" for i in range(6)],
                                       labels=list(range(6)), zero_division=0)
    cm = confusion_matrix(labels, ens_preds, labels=list(range(6))).tolist()

    result = {
        "n_models": len(args.exp_dirs),
        "exp_dirs": [str(p) for p in args.exp_dirs],
        "per_member_test_qwk": per_member_qwks,
        "per_member_mean": float(np.mean(per_member_qwks)),
        "per_member_std": float(np.std(per_member_qwks)),
        "ensemble_test_qwk": ens_qwk,
        "ensemble_test_acc": ens_acc,
        "ensemble_macro_f1": ens_macro_f1,
        "ensemble_weighted_f1": ens_weighted_f1,
        "ensemble_lift_over_mean": ens_qwk - float(np.mean(per_member_qwks)),
        "confusion_matrix": cm,
        "n_test_slides": len(labels),
        "slide_predictions": [
            {
                "slide_id": slide_names[i],
                "label": int(labels[i]),
                "ensemble_pred": int(ens_preds[i]),
                "ensemble_probs": [float(x) for x in avg_probs[i].tolist()],
            }
            for i in range(len(labels))
        ],
        "per_member_predictions": per_member_predictions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    (args.out.with_suffix(".txt")).write_text(report_txt)

    print()
    print("=" * 64)
    print(f"  ENSEMBLE ({len(args.exp_dirs)} seeds)")
    print(f"  Per-member test QWK: mean={np.mean(per_member_qwks):.4f} "
          f"std={np.std(per_member_qwks):.4f}")
    print(f"  Per-member range:    [{min(per_member_qwks):.4f}, {max(per_member_qwks):.4f}]")
    print(f"  Ensemble test QWK:   {ens_qwk:.4f}")
    print(f"  Ensemble test Acc:   {ens_acc*100:.2f}%")
    print(f"  Ensemble macro F1:   {ens_macro_f1:.4f}")
    print(f"  Ensemble weighted F1:{ens_weighted_f1:.4f}")
    print(f"  Lift over member mean: {ens_qwk - float(np.mean(per_member_qwks)):+.4f}")
    print(f"  Saved: {args.out}")
    print(f"  Report: {args.out.with_suffix('.txt')}")
    print("=" * 64)


if __name__ == "__main__":
    main()
