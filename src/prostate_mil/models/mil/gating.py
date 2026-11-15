"""Couche de gating personnalisee pour filtrer les artefacts de fond residuels.

Two gating modes:
- v1 (original): Mean/Std, one-sided (catches high-norm only)
- v2 (robust):   Median/MAD, double-sided (catches high AND low-norm outliers)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class CustomGatingLayer(nn.Module):
    """Applique une penalite dynamique aux tokens suspects d'etre des artefacts.

    Supports two modes:
    - mode="standard": Mean/Std one-sided gating (mu + z*sigma)
    - mode="robust":   Median/MAD double-sided gating (|x - median| / MAD > z)

    Legacy aliases "v1" (→ standard) and "v2" (→ robust) are accepted for
    backward compatibility with existing experiment config JSONs.
    """

    def __init__(
        self,
        variance_threshold: float,
        alpha: float = 1e-4,
        z_score_factor: float = 2.0,
        eps: float = 1e-8,
        mode: str = "standard",
    ) -> None:
        """Initialise la couche de gating.

        Args:
            variance_threshold: Seuil en dessous duquel un patch est considere
                comme peu texture (condition "faible variance spatiale").
            alpha: Facteur multiplicatif applique aux tokens artefacts.
            z_score_factor: Multiplicateur dans le critere de seuil.
                standard: mu + z * sigma
                robust:   |x - median| / MAD > z
            eps: Constante numerique pour eviter les divisions par zero.
            mode: "standard" (Mean/Std) or "robust" (Median/MAD).
                  Legacy aliases "v1" and "v2" are accepted.
        """

        super().__init__()

        if variance_threshold < 0:
            raise ValueError("`variance_threshold` doit etre >= 0.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("`alpha` doit etre compris entre 0 et 1.")
        if z_score_factor < 0:
            raise ValueError("`z_score_factor` doit etre >= 0.")
        if eps <= 0:
            raise ValueError("`eps` doit etre > 0.")
        _aliases = {"v1": "standard", "v2": "robust"}
        mode = _aliases.get(mode, mode)
        if mode not in ("standard", "robust"):
            raise ValueError(f"`mode` must be 'standard' or 'robust' (aliases 'v1'/'v2' also accepted), got '{mode}'.")

        self.variance_threshold = float(variance_threshold)
        self.alpha = float(alpha)
        self.z_score_factor = float(z_score_factor)
        self.eps = float(eps)
        self.mode = mode

    def forward(
        self,
        tokens: Tensor,
        spatial_variances: Tensor,
        return_metadata: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Filtre les tokens de patchs avant l'agregation par attention."""

        tokens, spatial_variances = self._validate_inputs(tokens, spatial_variances)
        token_norms = self._compute_token_norms(tokens)

        if self.mode == "standard":
            artifact_mask, stats = self._detect_v1(token_norms, spatial_variances)
        else:
            artifact_mask, stats = self._detect_v2(token_norms, spatial_variances)

        penalty_factors = self._build_penalty_factors(token_norms, artifact_mask)
        gated_tokens = tokens * penalty_factors.unsqueeze(-1)

        if not return_metadata:
            return gated_tokens

        metadata: dict[str, Tensor] = {
            "token_norms": token_norms,
            "artifact_mask": artifact_mask,
            "penalty_factors": penalty_factors,
            "artifact_ratio": artifact_mask.float().mean().unsqueeze(0),
            **stats,
        }
        return gated_tokens, metadata

    # ── V1: Original Mean/Std (one-sided) ────────────────────────────

    def _detect_v1(
        self, token_norms: Tensor, spatial_variances: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Original method: flag tokens with norm > mu + z*sigma AND low variance."""
        norm_mean = token_norms.mean()
        norm_std = token_norms.std(unbiased=False)
        norm_cutoff = norm_mean + self.z_score_factor * norm_std

        high_norm_mask = token_norms > (norm_cutoff + self.eps)
        low_variance_mask = spatial_variances < self.variance_threshold
        artifact_mask = high_norm_mask & low_variance_mask

        stats = {
            "norm_mean": norm_mean.unsqueeze(0),
            "norm_std": norm_std.unsqueeze(0),
            "norm_cutoff": norm_cutoff.unsqueeze(0),
        }
        return artifact_mask, stats

    # ── V2: Robust Median/MAD (double-sided) ─────────────────────────

    def _detect_v2(
        self, token_norms: Tensor, spatial_variances: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Robust method: flag tokens with |norm - median| / MAD > z AND low variance.

        Uses Median Absolute Deviation (MAD) instead of Std:
        - Immune to outlier contamination
        - Double-sided: catches both abnormally HIGH and LOW norms
        - 1.4826 scaling makes MAD consistent with Std for Gaussian data
        """
        norm_median = token_norms.median()
        abs_deviation = torch.abs(token_norms - norm_median)
        mad = abs_deviation.median() * 1.4826  # Scale to Gaussian consistency
        mad = torch.clamp(mad, min=self.eps)
 
        # Robust z-scores (double-sided)
        robust_z = abs_deviation / mad
        outlier_mask = robust_z > self.z_score_factor

        low_variance_mask = spatial_variances < self.variance_threshold
        artifact_mask = outlier_mask & low_variance_mask

        stats = {
            "norm_median": norm_median.unsqueeze(0),
            "norm_mad": mad.unsqueeze(0),
            "norm_cutoff_high": (norm_median + self.z_score_factor * mad).unsqueeze(0),
            "norm_cutoff_low": (norm_median - self.z_score_factor * mad).unsqueeze(0),
        }
        return artifact_mask, stats

    # ── Shared utilities ─────────────────────────────────────────────

    def _validate_inputs(
        self, tokens: Tensor, spatial_variances: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Valide les shapes et harmonise les tenseurs d'entree."""

        if tokens.ndim != 2:
            raise ValueError(
                f"`tokens` doit avoir 2 dimensions [N, D], recu {tuple(tokens.shape)}."
            )
        if tokens.shape[0] == 0:
            raise ValueError("`tokens` ne peut pas etre vide.")

        if spatial_variances.ndim == 2 and spatial_variances.shape[1] == 1:
            spatial_variances = spatial_variances.squeeze(1)
        elif spatial_variances.ndim != 1:
            raise ValueError(
                "`spatial_variances` doit avoir la forme [N] ou [N, 1]."
            )

        if spatial_variances.shape[0] != tokens.shape[0]:
            raise ValueError(
                "Le nombre de variances spatiales doit correspondre au nombre de tokens."
            )

        if not torch.is_floating_point(tokens):
            tokens = tokens.float()

        spatial_variances = spatial_variances.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, spatial_variances

    def _compute_token_norms(self, tokens: Tensor) -> Tensor:
        """Calcule la norme L2 de chaque token sur la dimension des features."""
        return torch.linalg.vector_norm(tokens, ord=2, dim=1)

    def _build_penalty_factors(
        self, reference: Tensor, artifact_mask: Tensor
    ) -> Tensor:
        """Construit un vecteur de poids multiplicatifs de forme `[N]`."""
        ones = torch.ones_like(reference)
        alpha_values = torch.full_like(reference, fill_value=self.alpha)
        return torch.where(artifact_mask, alpha_values, ones)

    def extra_repr(self) -> str:
        """Resume les hyperparametres dans `print(module)`."""
        return (
            f"mode={self.mode}, "
            f"variance_threshold={self.variance_threshold}, "
            f"alpha={self.alpha}, "
            f"z_score_factor={self.z_score_factor}"
        )


def summarize_gating_output(metadata: dict[str, Tensor]) -> dict[str, Any]:
    """Convertit les sorties de debug du gating en resume Python lisible."""

    summary = {
        "artifact_ratio": float(metadata["artifact_ratio"].item()),
        "num_artifacts": int(metadata["artifact_mask"].sum().item()),
    }

    # V1 fields
    if "norm_mean" in metadata:
        summary["norm_mean"] = float(metadata["norm_mean"].item())
        summary["norm_std"] = float(metadata["norm_std"].item())
        summary["norm_cutoff"] = float(metadata["norm_cutoff"].item())

    # V2 fields
    if "norm_median" in metadata:
        summary["norm_median"] = float(metadata["norm_median"].item())
        summary["norm_mad"] = float(metadata["norm_mad"].item())
        summary["norm_cutoff_high"] = float(metadata["norm_cutoff_high"].item())
        summary["norm_cutoff_low"] = float(metadata["norm_cutoff_low"].item())

    return summary
