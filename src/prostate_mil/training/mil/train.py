"""Point d'entree haut niveau pour l'entrainement du modele MIL.

Features:
- Early stopping (patience-based on val QWK)
- Cosine annealing LR schedule
- Patch dropout (data augmentation)
- Auto-saves experiment results (config, logs, plots)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from prostate_mil.config.defaults import GatingConfig, TrainingConfig
from prostate_mil.models.backbones.backbone import get_embedding_dim
from prostate_mil.models.mil.model import CustomMILModel
from prostate_mil.training.mil.dataset import SlideEmbeddingDataset
from prostate_mil.training.mil.engine import train_one_epoch, validate_one_epoch
from prostate_mil.training.mil.losses import CoralLoss, FocalLoss, coral_predict
from prostate_mil.training.mil.splits import fixed_split_indices
from prostate_mil.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _generate_plots(history: dict, out_dir: Path) -> None:
    """Generate training plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
    except ImportError:
        print("  ⚠️  matplotlib not installed, skipping plots")
        return

    epochs = np.array(history["epochs"])
    COLORS = {"train": "#2196F3", "val": "#FF5722", "best": "#4CAF50"}
    best_epoch = history["best_epoch"]

    # Plot 1: Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], color=COLORS["train"], lw=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], color=COLORS["val"], lw=2, label="Val Loss")
    ax.axvline(x=best_epoch, color=COLORS["best"], ls="--", alpha=0.7, label=f"Best (epoch {best_epoch})")
    if best_epoch < len(epochs):
        ax.axvspan(best_epoch, epochs[-1], alpha=0.06, color="red")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training & Validation Loss", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close(fig)

    # Plot 2: QWK
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_qwk"], color=COLORS["train"], lw=2, label="Train QWK")
    ax.plot(epochs, history["val_qwk"], color=COLORS["val"], lw=2, label="Val QWK")
    best_qwk = history["best_qwk"]
    ax.axhline(y=best_qwk, color=COLORS["best"], ls="--", alpha=0.7, label=f"Best QWK = {best_qwk:.4f}")
    ax.scatter([best_epoch], [best_qwk], color=COLORS["best"], s=200, zorder=5, marker="*")
    ax.axhline(y=0.70, color="gray", ls=":", alpha=0.4)
    ax.text(2, 0.705, "Inter-pathologist κ ≈ 0.70", fontsize=9, color="gray")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Quadratic Weighted Kappa", fontsize=13)
    ax.set_title("QWK Curves", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_ylim(0.65, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "qwk_curves.png", dpi=200)
    plt.close(fig)

    # Plot 3: Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_acc"], color=COLORS["train"], lw=2, label="Train Acc")
    ax.plot(epochs, history["val_acc"], color=COLORS["val"], lw=2, label="Val Acc")
    ax.axvline(x=best_epoch, color=COLORS["best"], ls="--", alpha=0.7, label=f"Best (epoch {best_epoch})")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Classification Accuracy", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curves.png", dpi=200)
    plt.close(fig)

    # Plot 4: Generalization gap
    fig, ax = plt.subplots(figsize=(10, 6))
    gap_loss = np.array(history["val_loss"]) - np.array(history["train_loss"])
    gap_qwk = np.array(history["train_qwk"]) - np.array(history["val_qwk"])
    ax.plot(epochs, gap_loss, color="#9C27B0", lw=2, label="Val Loss − Train Loss")
    ax.plot(epochs, gap_qwk, color="#FF9800", lw=2, label="Train QWK − Val QWK")
    ax.axhline(y=0, color="gray", ls="-", alpha=0.3)
    ax.axvline(x=best_epoch, color=COLORS["best"], ls="--", alpha=0.7, label="Best Model")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Gap", fontsize=13)
    ax.set_title("Generalization Gap", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap.png", dpi=200)
    plt.close(fig)

    print(f"  📊 Plots saved to {out_dir}")


def run_training(
    embeddings_dir: str | Path,
    labels_csv: str | Path,
    training_config: TrainingConfig | None = None,
    gating_config: GatingConfig | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    device: str = "auto",
    use_gating: bool = True,
    gating_mode: str = "robust",
    experiment_name: str = "",
    dropout: float = 0.25,
    patch_dropout: float = 0.0,
    early_stopping_patience: int = 0,
    use_cosine_lr: bool = False,
    aggregator: str = "abmil",
    backbone_name: str = "histoencoder",
    accumulation_steps: int = 8,
    transmil_layers: int = 2,
    init_seed: int = 0,
    gating_alpha: float = 1e-4,
    split_csv: str | Path | None = None,
) -> Path:
    """Full training loop with experiment tracking."""

    tc = training_config or TrainingConfig()
    gc = gating_config or GatingConfig()
    gc.input_dim = get_embedding_dim(backbone_name)

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        device if device != "auto" else "cpu"
    )

    # ── Experiment directory ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if not experiment_name:
        gating_tag = "gated" if use_gating else "nogating"
        experiment_name = f"{timestamp}_{backbone_name}_{aggregator}_{gating_tag}"
    exp_dir = ensure_dir(Path("outputs/experiments") / backbone_name / aggregator / experiment_name)

    # ── Dataset + split ──────────────────────────────────────────────
    full_dataset = SlideEmbeddingDataset(
        embeddings_dir=embeddings_dir, labels_csv=labels_csv,
    )
    train_indices, val_indices, test_indices = fixed_split_indices(
        full_dataset, val_ratio=val_ratio, test_ratio=test_ratio, split_csv=split_csv,
    )
    n_train, n_val, n_test = len(train_indices), len(val_indices), len(test_indices)

    # Seed model init + training randomness AFTER the split is built.
    # The split is fixed (CSV or its own seed 42), so varying
    # init_seed produces ensemble members that see the IDENTICAL split
    # but different weight inits, shuffle orders, and dropout masks.
    import random
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    train_loader = DataLoader(
        Subset(full_dataset, train_indices), batch_size=1, shuffle=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_indices), batch_size=1, shuffle=False,
    ) if len(val_indices) > 0 else None
    test_loader = DataLoader(
        Subset(full_dataset, test_indices), batch_size=1, shuffle=False,
    ) if len(test_indices) > 0 else None

    # ── Model ────────────────────────────────────────────────────────
    mode = "WITH GATING" if use_gating else "NO GATING (baseline)"
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"

    ordinal = tc.loss_type == "coral"
    model = CustomMILModel(
        input_dim=gc.input_dim,
        attention_dim=gc.attention_dim,
        num_classes=tc.num_classes,
        variance_threshold=gc.variance_threshold,
        alpha=gating_alpha,
        use_gating=use_gating,
        gating_mode=gating_mode,
        dropout=dropout,
        patch_dropout=patch_dropout,
        aggregator=aggregator,
        transmil_layers=transmil_layers,
        ordinal=ordinal,
    ).to(dev)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Save experiment config
    config = {
        "experiment_name": experiment_name,
        "backbone": f"{backbone_name} ({gc.input_dim}d)",
        "gating": use_gating,
        "gating_mode": gating_mode if use_gating else "N/A",
        "gating_alpha": gating_alpha if use_gating else "N/A",
        "dropout": dropout,
        "patch_dropout": patch_dropout,
        "early_stopping_patience": early_stopping_patience,
        "cosine_lr": use_cosine_lr,
        "epochs": tc.epochs,
        "learning_rate": tc.learning_rate,
        "weight_decay": tc.weight_decay,
        "loss_type": tc.loss_type,
        "focal_gamma": tc.focal_gamma if tc.loss_type == "focal" else "N/A",
        "accumulation_steps": accumulation_steps,
        "transmil_layers": transmil_layers if aggregator == "transmil" else "N/A",
        "variance_threshold": gc.variance_threshold,
        "z_score_factor": gc.z_score_factor,
        "train_slides": n_train,
        "val_slides": n_val,
        "test_slides": n_test,
        "parameters": param_count,
        "device": gpu_name,
        "init_seed": init_seed,
        "split_seed": 42,
        "split_csv": str(split_csv) if split_csv else None,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Pretty header ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  🧠 PHASE 3: MIL TRAINING")
    print("=" * 72)
    print(f"  Experiment:     {experiment_name}")
    print(f"  Mode:           {mode}" + (f" ({gating_mode})" if use_gating else ""))
    print(f"  Device:         {gpu_name}")
    print(f"  Train/Val/Test: {n_train} / {n_val} / {n_test}")
    print(f"  Epochs:         {tc.epochs}")
    loss_label = "CORAL (ordinal)" if ordinal else f"Focal (gamma={tc.focal_gamma})"
    print(f"  Loss:           {loss_label}")
    print(f"  LR:             {tc.learning_rate}" + (" + cosine" if use_cosine_lr else ""))
    print(f"  Weight decay:   {tc.weight_decay}")
    print(f"  Dropout:        {dropout}")
    print(f"  Patch dropout:  {patch_dropout}")
    if early_stopping_patience > 0:
        print(f"  Early stopping: patience={early_stopping_patience}")
    print(f"  Parameters:     {param_count:,}")
    print(f"  Results:        {exp_dir}")
    print("=" * 72)

    # ── Optimizer + scheduler ────────────────────────────────────────
    if ordinal:
        criterion = CoralLoss(num_classes=tc.num_classes)
    else:
        criterion = FocalLoss(gamma=tc.focal_gamma, alpha=tc.focal_alpha)
    optimizer = AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs) if use_cosine_lr else None

    # ── Training loop ────────────────────────────────────────────────
    best_val_qwk = -1.0
    best_epoch = 0
    patience_counter = 0
    best_path = exp_dir / "best_model.pt"

    history = {
        "epochs": [], "train_loss": [], "train_acc": [], "train_qwk": [],
        "val_loss": [], "val_acc": [], "val_qwk": [], "lr": [],
    }

    print()
    print(f"         │{'── Train ──':^32}│{'──── Val ────':^26}│")
    header = (
        f"  {'Epoch':>5}  │"
        f"  {'Loss':>7}  {'Acc':>6}  {'QWK':>6}  {'Art%':>5}  │"
        f"  {'Loss':>7}  {'Acc':>6}  {'QWK':>6}  │  {'Best':>6}  {'Time':>8}"
    )
    print(header)
    print("  " + "─" * 5 + "──┼" + "─" * 32 + "┼" + "─" * 26 + "┼" + "─" * 20)

    start_time = time.time()

    for epoch in range(1, tc.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, dev,
            accumulation_steps=accumulation_steps, ordinal=ordinal,
        )
        val_m = validate_one_epoch(model, val_loader, criterion, dev, ordinal=ordinal) if val_loader else None

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        track_qwk = val_m.qwk if val_m else train_m.qwk

        # Record history
        history["epochs"].append(epoch)
        history["train_loss"].append(train_m.loss)
        history["train_acc"].append(train_m.accuracy * 100)
        history["train_qwk"].append(train_m.qwk)
        history["val_loss"].append(val_m.loss if val_m else 0)
        history["val_acc"].append(val_m.accuracy * 100 if val_m else 0)
        history["val_qwk"].append(val_m.qwk if val_m else 0)
        history["lr"].append(current_lr)

        # Best model
        is_best = track_qwk > best_val_qwk
        if is_best:
            best_val_qwk = track_qwk
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_qwk": best_val_qwk,
                "config": config,
            }, best_path)
        else:
            patience_counter += 1

        elapsed = time.time() - start_time
        star = " ★" if is_best else "  "

        if val_m:
            print(
                f"  {epoch:3d}/{tc.epochs:2d}  │"
                f"  {train_m.loss:7.4f}  {train_m.accuracy*100:5.1f}%  {train_m.qwk:6.4f}  {train_m.artifact_ratio*100:4.1f}%  │"
                f"  {val_m.loss:7.4f}  {val_m.accuracy*100:5.1f}%  {val_m.qwk:6.4f}  │"
                f"  {best_val_qwk:6.4f}{star} {_format_time(elapsed):>7}"
            )

        # Early stopping
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"\n  ⏹️  Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
            break

    total_time = time.time() - start_time

    # ── Save history + generate plots ────────────────────────────────
    history["best_qwk"] = best_val_qwk
    history["best_epoch"] = best_epoch

    with open(exp_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    _generate_plots(history, exp_dir)

    # ── Final Test Evaluation ────────────────────────────────────────
    test_m = None
    if test_loader and best_val_qwk >= 0 and best_path.exists():
        print("\n  🧪 RUNNING FINAL EVALUATION ON HELD-OUT TEST SET...")
        # Load best weights
        checkpoint = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        test_m = validate_one_epoch(model, test_loader, criterion, dev, ordinal=ordinal)
        history["test_loss"] = test_m.loss
        history["test_acc"] = test_m.accuracy * 100
        history["test_qwk"] = test_m.qwk
        
        # Save updated history
        with open(exp_dir / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)
            
    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Experiment:  {experiment_name}")
    print(f"  Mode:        {mode}")
    print(f"  Best QWK:    {best_val_qwk:.4f} (epoch {best_epoch})")
    if test_m:
        print(f"  Test QWK:    {test_m.qwk:.4f}  <-- FINAL SCORE")
    print(f"  Total time:  {_format_time(total_time)}")
    print(f"  Checkpoint:  {best_path}")
    print(f"  Results:     {exp_dir}")
    print("=" * 72)

    return best_path


def run_kfold_training(
    embeddings_dir: str | Path,
    labels_csv: str | Path,
    n_folds: int = 5,
    training_config: TrainingConfig | None = None,
    gating_config: GatingConfig | None = None,
    device: str = "auto",
    use_gating: bool = True,
    gating_mode: str = "robust",
    experiment_name: str = "",
    dropout: float = 0.25,
    patch_dropout: float = 0.0,
    early_stopping_patience: int = 0,
    use_cosine_lr: bool = False,
    aggregator: str = "abmil",
    backbone_name: str = "histoencoder",
    accumulation_steps: int = 8,
    transmil_layers: int = 2,
    init_seed: int = 0,
    gating_alpha: float = 1e-4,
) -> Path:
    """Stratified K-Fold cross-validation with per-fold checkpoints.

    Every sample is used for validation exactly once across all folds.
    At the end, reports per-fold QWK, mean +/- std, and saves K models
    that can be ensembled for inference.
    """
    import random

    tc = training_config or TrainingConfig()
    gc = gating_config or GatingConfig()
    gc.input_dim = get_embedding_dim(backbone_name)

    dev = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        device if device != "auto" else "cpu"
    )

    # ── Experiment directory ─────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if not experiment_name:
        gating_tag = "gated" if use_gating else "nogating"
        experiment_name = f"{timestamp}_{backbone_name}_{aggregator}_{gating_tag}_{n_folds}fold"
    kfold_dir = ensure_dir(Path("outputs/experiments") / backbone_name / aggregator / experiment_name)

    # ── Dataset ─────────────────────────────────────────────────────
    full_dataset = SlideEmbeddingDataset(
        embeddings_dir=embeddings_dir, labels_csv=labels_csv,
    )
    n_total = len(full_dataset)
    labels = [full_dataset[i]["label"] for i in range(n_total)]

    mode = "WITH GATING" if use_gating else "NO GATING (baseline)"
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    ordinal = tc.loss_type == "coral"
    loss_label = "CORAL (ordinal)" if ordinal else f"Focal (gamma={tc.focal_gamma})"

    # ── K-Fold header ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  🧠 PHASE 3: MIL TRAINING — {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 72)
    print(f"  Experiment:     {experiment_name}")
    print(f"  Mode:           {mode}" + (f" ({gating_mode})" if use_gating else ""))
    print(f"  Device:         {gpu_name}")
    print(f"  Total slides:   {n_total}")
    print(f"  Folds:          {n_folds} (stratified)")
    print(f"  Epochs/fold:    {tc.epochs}")
    print(f"  Loss:           {loss_label}")
    print(f"  LR:             {tc.learning_rate}" + (" + cosine" if use_cosine_lr else ""))
    print(f"  Weight decay:   {tc.weight_decay}")
    print(f"  Dropout:        {dropout}")
    print(f"  Patch dropout:  {patch_dropout}")
    if early_stopping_patience > 0:
        print(f"  Early stopping: patience={early_stopping_patience}")
    print(f"  Results:        {kfold_dir}")
    print("=" * 72)

    # ── Stratified split ─────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results: list[dict] = []
    # Out-of-fold predictions: every slide gets one prediction from the
    # model that never trained on it. Aggregated into a single global QWK.
    oof_true: list[int] = []
    oof_pred: list[int] = []
    total_start = time.time()

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(range(n_total), labels)):
        fold_num = fold_idx + 1
        fold_dir = ensure_dir(kfold_dir / f"fold_{fold_num}")
        n_train, n_val = len(train_indices), len(val_indices)

        print()
        print(f"  ┌─── FOLD {fold_num}/{n_folds} ─── train={n_train} val={n_val} ───┐")

        # Seed per-fold for reproducibility
        fold_seed = init_seed + fold_idx
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        train_loader = DataLoader(
            Subset(full_dataset, train_indices.tolist()), batch_size=1, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_indices.tolist()), batch_size=1, shuffle=False,
        )

        # Fresh model per fold
        model = CustomMILModel(
            input_dim=gc.input_dim,
            attention_dim=gc.attention_dim,
            num_classes=tc.num_classes,
            variance_threshold=gc.variance_threshold,
            alpha=gating_alpha,
            use_gating=use_gating,
            gating_mode=gating_mode,
            dropout=dropout,
            patch_dropout=patch_dropout,
            aggregator=aggregator,
            transmil_layers=transmil_layers,
            ordinal=ordinal,
        ).to(dev)

        if fold_idx == 0:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Parameters:     {param_count:,}")

        if ordinal:
            criterion = CoralLoss(num_classes=tc.num_classes)
        else:
            criterion = FocalLoss(gamma=tc.focal_gamma, alpha=tc.focal_alpha)
        optimizer = AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=tc.epochs) if use_cosine_lr else None

        # Save fold config
        fold_config = {
            "experiment_name": experiment_name,
            "fold": fold_num,
            "n_folds": n_folds,
            "backbone": f"{backbone_name} ({gc.input_dim}d)",
            "aggregator": aggregator,
            "gating": use_gating,
            "gating_mode": gating_mode if use_gating else "N/A",
            "gating_alpha": gating_alpha if use_gating else "N/A",
            "dropout": dropout,
            "patch_dropout": patch_dropout,
            "early_stopping_patience": early_stopping_patience,
            "cosine_lr": use_cosine_lr,
            "epochs": tc.epochs,
            "learning_rate": tc.learning_rate,
            "weight_decay": tc.weight_decay,
            "loss_type": tc.loss_type,
            "focal_gamma": tc.focal_gamma if tc.loss_type == "focal" else "N/A",
            "accumulation_steps": accumulation_steps,
            "transmil_layers": transmil_layers if aggregator == "transmil" else "N/A",
            "variance_threshold": gc.variance_threshold,
            "z_score_factor": gc.z_score_factor,
            "train_slides": n_train,
            "val_slides": n_val,
            "device": gpu_name,
            "fold_seed": fold_seed,
            "split_seed": 42,
        }
        with open(fold_dir / "config.json", "w") as f:
            json.dump(fold_config, f, indent=2)

        # ── Training loop ────────────────────────────────────────────
        best_val_qwk = -1.0
        best_epoch = 0
        patience_counter = 0
        best_path = fold_dir / "best_model.pt"

        history = {
            "epochs": [], "train_loss": [], "train_acc": [], "train_qwk": [],
            "val_loss": [], "val_acc": [], "val_qwk": [], "lr": [],
        }

        print()
        print(f"         │{'── Train ──':^32}│{'──── Val ────':^26}│")
        header = (
            f"  {'Epoch':>5}  │"
            f"  {'Loss':>7}  {'Acc':>6}  {'QWK':>6}  {'Art%':>5}  │"
            f"  {'Loss':>7}  {'Acc':>6}  {'QWK':>6}  │  {'Best':>6}  {'Time':>8}"
        )
        print(header)
        print("  " + "─" * 5 + "──┼" + "─" * 32 + "┼" + "─" * 26 + "┼" + "─" * 20)

        fold_start = time.time()

        for epoch in range(1, tc.epochs + 1):
            train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, dev,
                accumulation_steps=accumulation_steps, ordinal=ordinal,
            )
            val_m = validate_one_epoch(model, val_loader, criterion, dev, ordinal=ordinal)

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler:
                scheduler.step()

            history["epochs"].append(epoch)
            history["train_loss"].append(train_m.loss)
            history["train_acc"].append(train_m.accuracy * 100)
            history["train_qwk"].append(train_m.qwk)
            history["val_loss"].append(val_m.loss)
            history["val_acc"].append(val_m.accuracy * 100)
            history["val_qwk"].append(val_m.qwk)
            history["lr"].append(current_lr)

            is_best = val_m.qwk > best_val_qwk
            if is_best:
                best_val_qwk = val_m.qwk
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_qwk": best_val_qwk,
                    "config": fold_config,
                }, best_path)
            else:
                patience_counter += 1

            elapsed = time.time() - fold_start
            star = " ★" if is_best else "  "
            print(
                f"  {epoch:3d}/{tc.epochs:2d}  │"
                f"  {train_m.loss:7.4f}  {train_m.accuracy*100:5.1f}%  {train_m.qwk:6.4f}  {train_m.artifact_ratio*100:4.1f}%  │"
                f"  {val_m.loss:7.4f}  {val_m.accuracy*100:5.1f}%  {val_m.qwk:6.4f}  │"
                f"  {best_val_qwk:6.4f}{star} {_format_time(elapsed):>7}"
            )

            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

        fold_time = time.time() - fold_start
        history["best_qwk"] = best_val_qwk
        history["best_epoch"] = best_epoch

        with open(fold_dir / "training_log.json", "w") as f:
            json.dump(history, f, indent=2)

        _generate_plots(history, fold_dir)

        # ── Out-of-fold predictions (best model on this fold's val set) ──
        checkpoint = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch["embeddings"].squeeze(0).to(dev)
                variances = batch["spatial_variances"].squeeze(0).to(dev)
                label = batch["label"].item()
                output = model(embeddings, variances)
                logits = output["logits"].unsqueeze(0)
                pred = coral_predict(logits).item() if ordinal else logits.argmax(dim=1).item()
                oof_true.append(label)
                oof_pred.append(pred)

        fold_results.append({
            "fold": fold_num,
            "best_val_qwk": best_val_qwk,
            "best_epoch": best_epoch,
            "train_slides": n_train,
            "val_slides": n_val,
            "time": _format_time(fold_time),
        })

        print(f"  └─── Fold {fold_num} done: best QWK = {best_val_qwk:.4f} "
              f"(epoch {best_epoch}) in {_format_time(fold_time)} ───┘")

    # ── K-Fold Summary ───────────────────────────────────────────────
    total_time = time.time() - total_start
    fold_qwks = [r["best_val_qwk"] for r in fold_results]
    mean_qwk = float(np.mean(fold_qwks))
    std_qwk = float(np.std(fold_qwks))

    # Global QWK from ALL out-of-fold predictions (every slide predicted
    # exactly once by the model that never trained on it).
    from prostate_mil.training.mil.engine import _quadratic_weighted_kappa
    global_qwk = _quadratic_weighted_kappa(oof_true, oof_pred)
    global_acc = sum(t == p for t, p in zip(oof_true, oof_pred)) / len(oof_true)

    summary = {
        "experiment_name": experiment_name,
        "n_folds": n_folds,
        "n_total_slides": n_total,
        "fold_results": fold_results,
        "mean_val_qwk": mean_qwk,
        "std_val_qwk": std_qwk,
        "min_val_qwk": float(min(fold_qwks)),
        "max_val_qwk": float(max(fold_qwks)),
        "global_oof_qwk": global_qwk,
        "global_oof_acc": global_acc,
        "total_time": _format_time(total_time),
    }
    with open(kfold_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 72)
    print(f"  ✅ {n_folds}-FOLD CROSS-VALIDATION COMPLETE")
    print("=" * 72)
    print(f"  Experiment:     {experiment_name}")
    print(f"  Mode:           {mode}")
    for r in fold_results:
        print(f"    Fold {r['fold']}: QWK = {r['best_val_qwk']:.4f} "
              f"(epoch {r['best_epoch']}, {r['time']})")
    print(f"  ─────────────────────────────────────────")
    print(f"  Mean fold QWK:  {mean_qwk:.4f} ± {std_qwk:.4f}")
    print(f"  Range:          [{min(fold_qwks):.4f}, {max(fold_qwks):.4f}]")
    print(f"  ─────────────────────────────────────────")
    print(f"  GLOBAL QWK:     {global_qwk:.4f}  <-- FINAL SCORE ({n_total} slides)")
    print(f"  GLOBAL Acc:     {global_acc*100:.2f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total time:     {_format_time(total_time)}")
    print(f"  Results:        {kfold_dir}")
    print("=" * 72)

    return kfold_dir
