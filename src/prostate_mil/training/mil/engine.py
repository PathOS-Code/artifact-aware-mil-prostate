"""Boucles d'entrainement et de validation du modele MIL.

Tracks QWK (Quadratic Weighted Kappa) alongside loss and accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from prostate_mil.training.mil.losses import coral_predict


def _quadratic_weighted_kappa(y_true: list[int], y_pred: list[int], n_classes: int = 6) -> float:
    """Compute QWK — THE metric for ISUP grading."""
    from collections import Counter

    if len(y_true) == 0:
        return 0.0

    # Build confusion matrix
    conf = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    # Weight matrix (quadratic)
    w = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i][j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    # Expected matrix (outer product of marginals)
    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    n = conf.sum()
    if n == 0:
        return 0.0
    expected = np.outer(hist_true, hist_pred) / n

    # QWK
    num = (w * conf).sum()
    den = (w * expected).sum()
    if den == 0:
        return 1.0
    return 1.0 - num / den


@dataclass(slots=True)
class EpochMetrics:
    """Conteneur de sorties pour le moteur d'entrainement."""

    loss: float
    accuracy: float
    qwk: float
    artifact_ratio: float  # Average % of tokens flagged as artifacts


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int = 8,
    ordinal: bool = False,
) -> EpochMetrics:
    """Effectue une passe d'entrainement sur un epoch complet."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true, all_pred = [], []
    artifact_ratios = []

    optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
        embeddings = batch["embeddings"].squeeze(0).to(device)
        variances = batch["spatial_variances"].squeeze(0).to(device)
        label = batch["label"].to(device)

        output = model(embeddings, variances)
        logits = output["logits"].unsqueeze(0)

        loss = criterion(logits, label)
        # Scale loss to normalize gradients over the accumulation steps
        (loss / accumulation_steps).backward()

        total += 1

        if total % accumulation_steps == 0 or total == len(dataloader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        predicted = coral_predict(logits) if ordinal else logits.argmax(dim=1)
        correct += (predicted == label).sum().item()

        all_true.append(label.item())
        all_pred.append(predicted.item())

        if "artifact_mask" in output:
            artifact_ratios.append(output["artifact_mask"].float().mean().item())

    return EpochMetrics(
        loss=total_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        qwk=_quadratic_weighted_kappa(all_true, all_pred),
        artifact_ratio=np.mean(artifact_ratios) if artifact_ratios else 0.0,
    )


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ordinal: bool = False,
) -> EpochMetrics:
    """Effectue une passe de validation sur un epoch complet."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true, all_pred = [], []
    artifact_ratios = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embeddings"].squeeze(0).to(device)
            variances = batch["spatial_variances"].squeeze(0).to(device)
            label = batch["label"].to(device)

            output = model(embeddings, variances)
            logits = output["logits"].unsqueeze(0)
            loss = criterion(logits, label)

            total_loss += loss.item()
            predicted = coral_predict(logits) if ordinal else logits.argmax(dim=1)
            correct += (predicted == label).sum().item()
            total += 1

            all_true.append(label.item())
            all_pred.append(predicted.item())

            if "artifact_mask" in output:
                artifact_ratios.append(output["artifact_mask"].float().mean().item())

    return EpochMetrics(
        loss=total_loss / max(total, 1),
        accuracy=correct / max(total, 1),
        qwk=_quadratic_weighted_kappa(all_true, all_pred),
        artifact_ratio=np.mean(artifact_ratios) if artifact_ratios else 0.0,
    )
