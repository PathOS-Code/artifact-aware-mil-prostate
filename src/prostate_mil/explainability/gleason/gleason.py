"""Gleason pattern analysis from pixel-level masks (Radboud subset)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import openslide

# Radboud mask encoding
GLEASON_LABELS = {
    0: "background",
    1: "stroma",
    2: "benign",
    3: "gleason_3",
    4: "gleason_4",
    5: "gleason_5",
}

# BGR colors for each Gleason pattern (for heatmap rendering)
GLEASON_COLORS_BGR = {
    0: (180, 130, 70),    # background — muted blue
    1: (170, 170, 170),   # stroma — gray
    2: (0, 180, 0),       # benign — green
    3: (0, 220, 220),     # Gleason 3 — yellow
    4: (0, 140, 255),     # Gleason 4 — orange
    5: (0, 0, 220),       # Gleason 5 — red
}


def assign_gleason_per_patch(
    mask_path: Path,
    coords: np.ndarray,
    patch_size: int = 256,
    provider: str = "radboud",
) -> tuple[np.ndarray, dict[str, float]]:
    """Read the Gleason mask and assign a dominant pattern to each patch.

    Args:
        mask_path: Path to the Radboud/Karolinska mask TIFF.
        coords: [N, 2] array of (coord_x, coord_y) at level 0.
        patch_size: Patch size in level-0 pixels.
        provider: 'radboud' or 'karolinska'.

    Returns:
        dominant_patterns: [N] array of dominant Gleason values (0-5) per patch.
        area_breakdown: Dict with percentage of tissue area per pattern.
    """
    slide = openslide.OpenSlide(str(mask_path))
    n = len(coords)
    dominant_patterns = np.zeros(n, dtype=np.int32)

    # Count total tissue pixels per Gleason pattern across all patches
    global_counts = np.zeros(6, dtype=np.int64)

    is_karolinska = provider.lower() == "karolinska"

    for i in range(n):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        region = slide.read_region((x, y), 0, (patch_size, patch_size))
        mask_patch = np.array(region)[:, :, 0]  # single channel

        if is_karolinska:
            # Karolinska mapping: 0->0 (bg), 1->2 (benign), 2->3 (cancer generic)
            # We map cancer to '3' so it shows up as Gleason 3 in the generic logic
            temp_mask = np.zeros_like(mask_patch)
            temp_mask[mask_patch == 1] = 2
            temp_mask[mask_patch == 2] = 3
            mask_patch = temp_mask

        # Count pixels per pattern in this patch
        for v in range(6):
            count = int((mask_patch == v).sum())
            global_counts[v] += count

        # Dominant pattern: most common non-background, non-stroma value
        tissue_mask = mask_patch[(mask_patch >= 2)]
        if len(tissue_mask) > 0:
            values, counts_local = np.unique(tissue_mask, return_counts=True)
            dominant_patterns[i] = values[counts_local.argmax()]
        else:
            dominant_patterns[i] = 0  # no tissue

    slide.close()

    # Compute area breakdown (only tissue: values 2-5)
    tissue_total = global_counts[2:].sum()
    area_breakdown = {}
    for v in range(2, 6):
        label = GLEASON_LABELS[v]
        pct = float(global_counts[v] / tissue_total * 100) if tissue_total > 0 else 0.0
        area_breakdown[label] = round(pct, 2)

    return dominant_patterns, area_breakdown


def derive_gleason_score(area_breakdown: dict[str, float], provider: str = "radboud") -> dict:
    """Derive Gleason Score from area percentages.

    Returns dict with primary_pattern, secondary_pattern, gleason_score,
    and derived ISUP grade.
    """
    is_karolinska = provider.lower() == "karolinska"

    # Extract pattern percentages (only cancerous: G3, G4, G5)
    patterns = {
        3: area_breakdown.get("gleason_3", 0.0),
        4: area_breakdown.get("gleason_4", 0.0),
        5: area_breakdown.get("gleason_5", 0.0),
    }

    cancer_total = sum(patterns.values())
    if cancer_total == 0:
        return {
            "primary_pattern": 0,
            "secondary_pattern": 0,
            "gleason_score": "benign",
            "isup_from_mask": 0,
            "cancer_percentage": 0.0,
        }

    if is_karolinska:
        # Karolinska doesn't provide patterns, only binary cancer presence
        return {
            "primary_pattern": None,
            "secondary_pattern": None,
            "gleason_score": "N/A (Karolinska)",
            "isup_from_mask": None,
            "cancer_percentage": round(cancer_total, 2),
            "pattern_areas": {"cancer": round(cancer_total, 2)},
        }

    # Primary = most area, Secondary = second most (or same if only one pattern)
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_patterns[0][0]

    # Secondary: highest pattern that is present and != primary, else same as primary
    secondary = primary
    for pat, pct in sorted_patterns[1:]:
        if pct > 0:
            secondary = pat
            break

    gs = primary + secondary

    # ISUP mapping
    isup_map = {
        (3, 3): 1,
        (3, 4): 2,
        (4, 3): 3,
        (4, 4): 4,
        (3, 5): 4,
        (5, 3): 4,
        (4, 5): 5,
        (5, 4): 5,
        (5, 5): 5,
    }
    isup = isup_map.get((primary, secondary), 0)

    return {
        "primary_pattern": primary,
        "secondary_pattern": secondary,
        "gleason_score": f"{primary}+{secondary}",
        "gleason_sum": gs,
        "isup_from_mask": isup,
        "cancer_percentage": round(cancer_total, 2),
        "pattern_areas": {
            f"gleason_{k}": round(v, 2) for k, v in patterns.items() if v > 0
        },
    }


def build_gleason_heatmap(
    coords: np.ndarray,
    dominant_patterns: np.ndarray,
    canvas_shape: tuple[int, int],
    patch_size: int = 256,
    downsample_factor: float = 1.0,
) -> np.ndarray:
    """Build a color-coded Gleason pattern heatmap.

    Returns BGR [H, W, 3] image where each patch is colored by its
    dominant Gleason pattern.
    """
    h, w = canvas_shape
    canvas = np.full((h, w, 3), 220, dtype=np.uint8)  # light gray background
    patch_thumb = max(1, int(round(patch_size / downsample_factor)))

    for i in range(len(coords)):
        tx = int(round(coords[i, 0] / downsample_factor))
        ty = int(round(coords[i, 1] / downsample_factor))

        y0, x0 = max(0, ty), max(0, tx)
        y1 = min(h, ty + patch_thumb)
        x1 = min(w, tx + patch_thumb)

        if y1 <= y0 or x1 <= x0:
            continue

        color = GLEASON_COLORS_BGR.get(dominant_patterns[i], (170, 170, 170))
        canvas[y0:y1, x0:x1] = color

    return canvas


def build_combined_heatmap(
    coords: np.ndarray,
    attention_scores: np.ndarray,
    dominant_patterns: np.ndarray,
    canvas_shape: tuple[int, int],
    patch_size: int = 256,
    downsample_factor: float = 1.0,
) -> np.ndarray:
    """Build a heatmap combining Gleason pattern color + attention brightness.

    Each patch gets its Gleason color, with brightness modulated by attention
    weight — bright = high attention, dark = low attention.

    Returns BGR [H, W, 3] image.
    """
    h, w = canvas_shape
    canvas = np.full((h, w, 3), 220, dtype=np.uint8)
    patch_thumb = max(1, int(round(patch_size / downsample_factor)))

    # Percentile-clip then map to [0.3, 1.0] so the gradient is visible
    scores = attention_scores.astype(np.float32)
    scores -= scores.min()
    cap = np.percentile(scores, 95)
    if cap > 0:
        scores = np.clip(scores, 0, cap) / cap
    else:
        s_max = scores.max()
        if s_max > 0:
            scores = scores / s_max
    scores = 0.3 + 0.7 * scores  # map to [0.3, 1.0]

    for i in range(len(coords)):
        tx = int(round(coords[i, 0] / downsample_factor))
        ty = int(round(coords[i, 1] / downsample_factor))

        y0, x0 = max(0, ty), max(0, tx)
        y1 = min(h, ty + patch_thumb)
        x1 = min(w, tx + patch_thumb)

        if y1 <= y0 or x1 <= x0:
            continue

        color = np.array(GLEASON_COLORS_BGR.get(dominant_patterns[i], (170, 170, 170)),
                         dtype=np.float32)
        canvas[y0:y1, x0:x1] = (color * scores[i]).astype(np.uint8)

    return canvas
