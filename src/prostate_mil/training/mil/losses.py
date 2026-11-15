"""Fonctions de perte pour classification de grades ISUP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Focal Loss multi-classes pour gerer le desequilibre de classes.

    Supports:
    - Scalar alpha: uniform weight for all classes
    - Per-class alpha: tensor of shape [num_classes] for class-specific weights
      (useful for PANDA where ISUP 0 is overrepresented)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Store alpha as a registered buffer for device compatibility
        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor([alpha]))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Calcule la Focal Loss a partir de logits non normalises.

        Args:
            logits: [B, C] raw logits
            targets: [B] integer class indices
        """

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                # Scalar alpha — uniform weight
                loss = self.alpha * loss
            else:
                # Per-class alpha — gather weight for each target
                alpha_t = self.alpha.gather(0, targets)
                loss = alpha_t * loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class CoralLoss(nn.Module):
    """CORAL ordinal regression loss (Cao et al., 2020).

    Decomposes K-class ordinal classification into K-1 cumulative binary
    tasks: "is grade > k?" for k in 0..K-2.  Penalty naturally scales
    with distance — predicting ISUP 4 when truth is 5 triggers 1 wrong
    threshold, while predicting 0 triggers 5 wrong thresholds.

    Expects logits of shape [B, K-1] (raw, pre-sigmoid) and integer
    targets of shape [B] in {0, ..., K-1}.
    """

    def __init__(self, num_classes: int = 6, reduction: str = "mean") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute CORAL loss.

        Args:
            logits: [B, K-1] raw ordinal logits (pre-sigmoid)
            targets: [B] integer class indices in {0, ..., K-1}
        """
        num_thresholds = self.num_classes - 1
        # Build cumulative binary labels: [B, K-1]
        # For target=3 with K=6: [1, 1, 1, 0, 0]
        levels = torch.arange(num_thresholds, device=targets.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()

        # Binary cross-entropy per threshold, summed across thresholds
        loss = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none",
        )
        # Sum across K-1 thresholds, then reduce across batch
        loss = loss.sum(dim=1)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def coral_predict(logits: Tensor) -> Tensor:
    """Convert CORAL logits [B, K-1] to class predictions [B].

    Predicted class = number of thresholds exceeded (sigmoid > 0.5,
    i.e. raw logit > 0).
    """
    return (logits > 0.0).sum(dim=1).long()
