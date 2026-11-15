"""Segmentation tissu/fond sur thumbnails WSI."""

from __future__ import annotations

import cv2
import numpy as np


def compute_hsv_otsu_mask(thumbnail_rgb: np.ndarray) -> np.ndarray:
    """Construit un masque binaire tissu/fond par Otsu sur l'espace HSV.

    La segmentation est basee sur deux indices derives de HSV:
    - la saturation `S`, elevee sur le tissu colore et faible sur le fond blanc
    - l'inverse de la luminance `255 - V`, utile pour recuperer les zones sombres

    Le masque final est l'union logique de deux seuils d'Otsu calcules sur ces
    canaux derives, puis converti en image binaire `uint8`.
    """

    if thumbnail_rgb.ndim != 3 or thumbnail_rgb.shape[2] != 3:
        raise ValueError("`thumbnail_rgb` doit avoir la forme [H, W, 3].")

    thumbnail_rgb = np.asarray(thumbnail_rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    inverse_value = 255 - value

    _, saturation_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, inverse_value_mask = cv2.threshold(
        inverse_value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_or(saturation_mask, inverse_value_mask)

    # Elimine explicitement les regions de fond tres blanches.
    bright_background = (value > 245) & (saturation < 20)
    mask[bright_background] = 0

    return mask.astype(np.uint8)


def refine_tissue_mask(mask: np.ndarray, min_area: int = 128) -> np.ndarray:
    """Nettoie le masque brut via operations morphologiques simples."""

    if mask.ndim != 2:
        raise ValueError("`mask` doit etre une image 2D.")

    mask_uint8 = np.where(mask > 0, 255, 0).astype(np.uint8)

    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = np.ones((7, 7), dtype=np.uint8)

    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    filtered = np.zeros_like(cleaned)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label] = 255

    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, close_kernel)
    return filtered.astype(np.uint8)
