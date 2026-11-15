"""Generation de cartes d'attention spatialement alignees a la WSI."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def normalize_attention_scores(
    attention_scores: np.ndarray,
    clip_percentile: float = 95.0,
) -> np.ndarray:
    """Normalise les poids d'attention entre 0 et 255.

    Uses percentile clipping to handle the extreme skew typical of MIL
    attention (max/median often >50x). Without clipping, 90%+ of patches
    map to near-zero and the heatmap looks empty.

    Args:
        attention_scores: Raw attention weights [N].
        clip_percentile: Upper percentile for clipping (default 95).
            Values above this are clamped to 255 (saturated red).
            Set to 100.0 for no clipping (original linear behavior).
    """
    if attention_scores.size == 0:
        return attention_scores.astype(np.uint8)

    scores = attention_scores.astype(np.float32)
    scores -= scores.min()

    if clip_percentile < 100.0:
        cap = np.percentile(scores, clip_percentile)
        if cap > 0:
            scores = np.clip(scores, 0, cap)

    max_value = scores.max()
    if max_value > 0:
        scores /= max_value
    return (scores * 255.0).astype(np.uint8)


def build_attention_heatmap(
    patch_registry: pd.DataFrame,
    attention_scores: np.ndarray,
    canvas_shape: tuple[int, int],
    patch_size: int = 256,
    downsample_factor: float = 1.0,
) -> np.ndarray:
    """Projette les poids d'attention sur une grille 2D de niveau slide.

    Args:
        patch_registry: DataFrame avec colonnes ``coord_x``, ``coord_y``
            (coordonnees level-0 en pixels).
        attention_scores: Vecteur [N] de poids d'attention bruts.
        canvas_shape: (height, width) du canvas de sortie (taille thumbnail).
        patch_size: Taille des patchs en pixels level-0.
        downsample_factor: Rapport entre level-0 et le thumbnail.

    Returns:
        Heatmap uint8 mono-canal [H, W] avec valeurs dans [0, 255].
    """

    h, w = canvas_shape
    canvas = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    scores_norm = normalize_attention_scores(attention_scores).astype(np.float32)
    patch_thumb = max(1, int(round(patch_size / downsample_factor)))

    coords_x = patch_registry["coord_x"].values
    coords_y = patch_registry["coord_y"].values

    for i in range(len(scores_norm)):
        tx = int(round(coords_x[i] / downsample_factor))
        ty = int(round(coords_y[i] / downsample_factor))

        y0 = max(0, ty)
        x0 = max(0, tx)
        y1 = min(h, ty + patch_thumb)
        x1 = min(w, tx + patch_thumb)

        if y1 <= y0 or x1 <= x0:
            continue

        canvas[y0:y1, x0:x1] += scores_norm[i]
        counts[y0:y1, x0:x1] += 1.0

    mask = counts > 0
    canvas[mask] /= counts[mask]

    return canvas.astype(np.uint8)


def colorize_heatmap(
    heatmap_gray: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Applique une colormap OpenCV a une heatmap mono-canal [0, 255].

    Returns:
        Image BGR uint8 [H, W, 3].
    """

    return cv2.applyColorMap(heatmap_gray, colormap)


