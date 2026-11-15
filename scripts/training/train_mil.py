"""Script CLI pour la phase 3: entrainement MIL.

All experiments auto-save to outputs/experiments/<name>/
with config, training logs, plots, and best checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: MIL training on pre-computed embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)

    # Experiment
    parser.add_argument("--name", type=str, default="",
                        help="Experiment name. Auto-generated if empty.")
    parser.add_argument("--backbone", type=str, choices=["histoencoder", "virchow2"], default="histoencoder",
                        help="Backbone used for embeddings. Used for folder organization.")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--loss", type=str, choices=["focal", "coral"], default="focal",
                        help="Loss function: 'focal' (categorical) or 'coral' (ordinal regression).")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--split-csv", type=Path, default=None,
                        help="Fixed split file (slide_id,split), e.g. splits/panda_fixed_split.csv. "
                             "Overrides --val-ratio/--test-ratio. Ignored with --kfold.")
    parser.add_argument("--device", type=str, default="auto")

    # Regularization
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Classifier head dropout (0.0-1.0)")
    parser.add_argument("--patch-dropout", type=float, default=0.0,
                        help="Random patch drop rate (0.0-0.5). Data augmentation.")
    parser.add_argument("--accumulation-steps", type=int, default=8,
                        help="Simulates larger batch size by accumulating gradients before stepping param updates.")
    parser.add_argument("--early-stopping", type=int, default=0,
                        help="Stop if no improvement for N epochs. 0=disabled.")
    parser.add_argument("--cosine-lr", action="store_true",
                        help="Use cosine annealing LR schedule.")

    # Ablation
    parser.add_argument("--no-gating", action="store_true",
                        help="Disable gating layer (ablation baseline).")
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="Gating attenuation factor for flagged tokens. "
                             "Use 1e-4 for full gate (default), 1.0 for mask-only (no feature attenuation).")
    parser.add_argument("--gating-mode", type=str, choices=["standard", "robust"], default="robust",
                        help="Which gating statistic to use: standard (mean/std) or robust (median/MAD).")
    parser.add_argument("--aggregator", type=str, choices=["abmil", "transmil"], default="abmil",
                        help="MIL aggregator mechanism.")
    parser.add_argument("--transmil-layers", type=int, default=2,
                        help="Number of Transformer layers to use if aggregator=transmil.")

    # Cross-validation
    parser.add_argument("--kfold", type=int, default=0,
                        help="Run stratified K-fold CV instead of single split. "
                             "0=disabled (default single train/val/test split). "
                             "E.g. --kfold 5 for 5-fold CV.")

    # Reproducibility / ensembling
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for model init + dataloader shuffle. "
                             "Data split seed is fixed at 42 independently, "
                             "so varying --seed produces ensemble members on the SAME split.")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from prostate_mil.config.defaults import TrainingConfig
    from prostate_mil.training.mil.train import run_kfold_training, run_training

    args = parse_args()

    config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        loss_type=args.loss,
        epochs=args.epochs,
    )

    shared_kwargs = dict(
        embeddings_dir=args.embeddings_dir,
        labels_csv=args.labels_csv,
        training_config=config,
        device=args.device,
        use_gating=not args.no_gating,
        gating_mode=args.gating_mode,
        gating_alpha=args.alpha,
        experiment_name=args.name,
        dropout=args.dropout,
        patch_dropout=args.patch_dropout,
        early_stopping_patience=args.early_stopping,
        use_cosine_lr=args.cosine_lr,
        aggregator=args.aggregator,
        backbone_name=args.backbone,
        accumulation_steps=args.accumulation_steps,
        transmil_layers=args.transmil_layers,
        init_seed=args.seed,
    )

    if args.kfold > 0:
        result_path = run_kfold_training(
            n_folds=args.kfold,
            **shared_kwargs,
        )
        print(f"\nK-fold complete. Results: {result_path}")
    else:
        result_path = run_training(
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            split_csv=args.split_csv,
            **shared_kwargs,
        )
        print(f"\nTraining complete. Best model: {result_path}")


if __name__ == "__main__":
    main()
