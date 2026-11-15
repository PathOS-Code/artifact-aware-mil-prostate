"""Normalisation de couleur Macenko pour patchs histopathologiques.

Deux implementations:
- MacenkoNormalizerWrapper  : CPU (numpy) pour Phase 1
- TorchMacenkoNormalizerGPU : GPU (torch) pour Phase 2 — contourne le bug
  Intel MKL SGELSY en utilisant torch.linalg.pinv au lieu de lstsq.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchstain.numpy.normalizers import NumpyMacenkoNormalizer

logger = logging.getLogger(__name__)


class MacenkoNormalizerWrapper:
    """Facade autour de `torchstain` pour normaliser les patchs (CPU)."""

    def __init__(self) -> None:
        self.normalizer = NumpyMacenkoNormalizer()
        self.is_fitted = False

    def fit(self, target_patch_rgb: np.ndarray) -> None:
        validated_patch = self._validate_patch(target_patch_rgb)
        self.normalizer.fit(validated_patch)
        self.is_fitted = True

    def fit_from_path(self, image_path: str | Path) -> None:
        image = Image.open(image_path).convert("RGB")
        self.fit(np.asarray(image, dtype=np.uint8))

    def transform(self, patch_rgb: np.ndarray) -> np.ndarray:
        validated_patch = self._validate_patch(patch_rgb)
        if validated_patch.std() < 5:
            return validated_patch
        try:
            normalized, _, _ = self.normalizer.normalize(validated_patch, stains=False)
            return np.asarray(normalized, dtype=np.uint8)
        except Exception:
            return validated_patch

    def _validate_patch(self, patch_rgb: np.ndarray) -> np.ndarray:
        if patch_rgb.ndim != 3 or patch_rgb.shape[2] != 3:
            raise ValueError("`patch_rgb` doit avoir la forme [H, W, 3].")
        return np.asarray(patch_rgb, dtype=np.uint8)


# ---------------------------------------------------------------------------
# GPU Macenko — custom implementation to avoid Intel MKL SGELSY bug
# ---------------------------------------------------------------------------


def _cov(X: torch.Tensor) -> torch.Tensor:
    """Covariance matrix for [D, N] matrix."""
    D, N = X.shape
    mean = X.mean(dim=1, keepdim=True)
    X_centered = X - mean
    return (X_centered @ X_centered.T) / max(N - 1, 1)


def _percentile(t: torch.Tensor, q: float) -> torch.Tensor:
    """Percentile on a 1-D tensor (GPU-safe)."""
    k = 1 + round(0.01 * q * (t.numel() - 1))
    k = max(1, min(k, t.numel()))
    return t.view(-1).kthvalue(k).values


class TorchMacenkoNormalizerWrapper:
    """Macenko normalisation entierement sur GPU.

    Contourne le bug Intel MKL SGELSY en remplacant torch.linalg.lstsq
    par torch.linalg.pinv (pseudo-inverse), ce qui utilise cuSOLVER
    directement sur le GPU au lieu de MKL sur le CPU.
    """

    # Stain reference vectors (H&E) and max concentrations
    HERef = torch.tensor([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
    maxCRef = torch.tensor([1.9705, 1.0308])

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.HERef = self.HERef.to(self.device)
        self.maxCRef = self.maxCRef.to(self.device)

    def _is_tissue_patch(self, img: torch.Tensor) -> bool:
        """Quick check: does this patch contain enough stained tissue?

        Args:
            img: [3, H, W] float tensor in [0, 1]
        """
        # Too bright (mostly white background)
        if img.mean() > 0.9:
            return False
        # Too flat (no contrast at all)
        if img.std() < 0.03:
            return False
        return True

    def _normalize_single(self, img: torch.Tensor, Io: int = 240,
                          alpha: float = 1, beta: float = 0.15) -> torch.Tensor:
        """Normalize a single [3, H, W] uint8 image on GPU.

        Returns [H, W, 3] int tensor.
        """
        c, h, w = img.shape

        # --- OD conversion ---
        I_hwc = img.permute(1, 2, 0).float()  # [H, W, 3]
        OD = -torch.log((I_hwc.reshape(-1, 3) + 1.0) / Io)

        # Remove transparent pixels
        mask = ~torch.any(OD < beta, dim=1)
        ODhat = OD[mask]

        if ODhat.shape[0] < 10:
            # Not enough tissue pixels — return original
            return img.permute(1, 2, 0).int()

        # --- Eigenvectors of covariance ---
        cov_mat = _cov(ODhat.T)
        try:
            _, eigvecs = torch.linalg.eigh(cov_mat)
        except Exception:
            return img.permute(1, 2, 0).int()

        eigvecs = eigvecs[:, [1, 2]]

        # --- Find H&E vectors ---
        That = ODhat @ eigvecs
        phi = torch.atan2(That[:, 1], That[:, 0])

        if phi.numel() == 0:
            return img.permute(1, 2, 0).int()

        minPhi = _percentile(phi, alpha)
        maxPhi = _percentile(phi, 100 - alpha)

        vMin = (eigvecs @ torch.stack([torch.cos(minPhi), torch.sin(minPhi)])).unsqueeze(1)
        vMax = (eigvecs @ torch.stack([torch.cos(maxPhi), torch.sin(maxPhi)])).unsqueeze(1)

        HE = torch.where(vMin[0] > vMax[0],
                         torch.cat([vMin, vMax], dim=1),
                         torch.cat([vMax, vMin], dim=1))

        # --- Concentrations via pseudo-inverse (NO lstsq = NO MKL SGELSY) ---
        Y = OD.T  # [3, N]
        C = torch.linalg.pinv(HE) @ Y  # [2, N]

        maxC = torch.stack([_percentile(C[0, :], 99), _percentile(C[1, :], 99)])
        maxC = torch.clamp(maxC, min=1e-6)  # avoid division by zero

        # Normalize concentrations
        C = C * (self.maxCRef / maxC).unsqueeze(-1)

        # Reconstruct
        Inorm = Io * torch.exp(-self.HERef @ C)
        Inorm = torch.clamp(Inorm, 0, 255)
        Inorm = Inorm.T.reshape(h, w, c).int()

        return Inorm

    def transform(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Normalise un batch [B, 3, H, W] (0.0-1.0) -> [B, 3, H, W] (0.0-1.0).

        Runs entirely on GPU. No Intel MKL calls.
        """
        B, C, H, W = batch_tensor.shape
        normalized_list = []

        for i in range(B):
            original = batch_tensor[i].to(self.device)

            # Skip non-tissue patches entirely
            if not self._is_tissue_patch(original):
                normalized_list.append(original)
                continue

            try:
                # Convert from [0,1] float to [0,255] uint8 for Macenko
                img_uint8 = (original * 255).to(torch.uint8)
                result = self._normalize_single(img_uint8)  # [H, W, 3] int
                # Back to [3, H, W] float [0,1]
                result_chw = result.permute(2, 0, 1).float() / 255.0
                normalized_list.append(result_chw)
            except Exception:
                normalized_list.append(original)

        return torch.stack(normalized_list).to(self.device)