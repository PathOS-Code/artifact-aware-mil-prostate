"""Modele MIL complet: gating + ABMIL + classification ISUP.

Supports:
- --no-gating mode for ablation study
- Configurable dropout rate
- Patch dropout (data augmentation: randomly drops patches from the bag)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import ABMILAttention, TransMILAttention
from .gating import CustomGatingLayer


class CustomMILModel(nn.Module):
    """Assemble les briques centrales de la phase 3 et de la phase 4."""

    def __init__(
        self,
        input_dim: int = 384,
        attention_dim: int = 256,
        num_classes: int = 6,
        variance_threshold: float = 122.0,
        alpha: float = 1e-4,
        use_gating: bool = True,
        dropout: float = 0.25,
        patch_dropout: float = 0.0,
        gating_mode: str = "robust",
        aggregator: str = "abmil",
        transmil_layers: int = 2,
        ordinal: bool = False,
    ) -> None:
        super().__init__()
        self.use_gating = use_gating
        self.patch_dropout = patch_dropout
        self.ordinal = ordinal

        if use_gating:
            self.gating = CustomGatingLayer(
                variance_threshold=variance_threshold,
                alpha=alpha,
                mode=gating_mode,
            )
        else:
            self.gating = None

        if aggregator == "transmil":
            self.attention = TransMILAttention(
                input_dim=input_dim,
                attention_dim=attention_dim,
                n_layers=transmil_layers,
            )
        else:
            self.attention = ABMILAttention(
                input_dim=input_dim,
                attention_dim=attention_dim,
            )

        # CORAL ordinal head outputs K-1 cumulative threshold logits;
        # standard categorical head outputs K class logits.
        output_dim = num_classes - 1 if ordinal else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, tokens: Tensor, variances: Tensor) -> dict[str, Tensor]:
        """Retourne logits, embedding slide-level et poids d'attention."""

        # Patch dropout: randomly drop patches during training
        if self.training and self.patch_dropout > 0.0 and tokens.shape[0] > 10:
            n = tokens.shape[0]
            keep = int(n * (1.0 - self.patch_dropout))
            keep = max(keep, 5)  # Keep at least 5 patches
            idx = torch.randperm(n, device=tokens.device)[:keep]
            idx = idx.sort().values  # Maintain order
            tokens = tokens[idx]
            variances = variances[idx]

        if self.use_gating and self.gating is not None:
            gated_tokens, gating_metadata = self.gating(
                tokens=tokens,
                spatial_variances=variances,
                return_metadata=True,
            )
            artifact_mask = gating_metadata["artifact_mask"]
            penalty_factors = gating_metadata["penalty_factors"]
        else:
            gated_tokens = tokens
            artifact_mask = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
            penalty_factors = torch.ones(tokens.shape[0], device=tokens.device)

        slide_embedding, attention_weights = self.attention(
            tokens=gated_tokens,
            mask=artifact_mask if self.use_gating else None,
            return_attention=True,
        )
        logits = self.classifier(slide_embedding)

        return {
            "logits": logits,
            "slide_embedding": slide_embedding,
            "attention_weights": attention_weights,
            "gated_tokens": gated_tokens,
            "artifact_mask": artifact_mask,
            "penalty_factors": penalty_factors,
        }
