"""Overlay de heatmap sur thumbnail WSI pour sortie clinique."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_heatmap_on_thumbnail(
    thumbnail_rgb: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Superpose une heatmap coloree a un thumbnail de la lame.

    Args:
        thumbnail_rgb: Thumbnail WSI en RGB [H, W, 3].
        heatmap_bgr: Heatmap colorisee en BGR [H', W', 3].
        alpha: Opacite de la heatmap (0 = invisible, 1 = opaque).

    Returns:
        Image RGB uint8 [H, W, 3] avec la heatmap superposee.
    """

    h, w = thumbnail_rgb.shape[:2]

    if heatmap_bgr.shape[:2] != (h, w):
        heatmap_bgr = cv2.resize(heatmap_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    thumbnail_bgr = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(heatmap_bgr, alpha, thumbnail_bgr, 1 - alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
