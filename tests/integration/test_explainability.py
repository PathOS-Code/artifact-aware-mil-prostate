"""Tests unitaires — Phase 4 : heatmap, colorisation et overlay.

Execution :
    python -m pytest tests/test_explainability.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prostate_mil.explainability.attention.heatmap import (
    build_attention_heatmap,
    colorize_heatmap,
    normalize_attention_scores,
)
from prostate_mil.explainability.attention.overlay import overlay_heatmap_on_thumbnail


# ---------------------------------------------------------------------------
# normalize_attention_scores
# ---------------------------------------------------------------------------

class TestNormalizeAttentionScores:

    def test_range_0_255(self) -> None:
        scores = np.array([0.1, 0.5, 0.9, 1.0])
        result = normalize_attention_scores(scores)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_empty_array(self) -> None:
        result = normalize_attention_scores(np.array([]))
        assert result.size == 0

    def test_uniform_scores(self) -> None:
        result = normalize_attention_scores(np.array([0.5, 0.5, 0.5]))
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# build_attention_heatmap
# ---------------------------------------------------------------------------

class TestBuildAttentionHeatmap:

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        registry = pd.DataFrame({
            "coord_x": [0, 256, 512, 0, 256],
            "coord_y": [0, 0, 0, 256, 256],
        })
        scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        return registry, scores

    def test_output_shape(self, sample_data) -> None:
        registry, scores = sample_data
        canvas = build_attention_heatmap(
            registry, scores, canvas_shape=(100, 100), patch_size=256, downsample_factor=8.0,
        )
        assert canvas.shape == (100, 100)
        assert canvas.dtype == np.uint8

    def test_non_zero_where_patches_exist(self, sample_data) -> None:
        registry, scores = sample_data
        canvas = build_attention_heatmap(
            registry, scores, canvas_shape=(100, 100), patch_size=256, downsample_factor=8.0,
        )
        assert canvas.max() > 0

    def test_empty_registry(self) -> None:
        registry = pd.DataFrame({"coord_x": [], "coord_y": []})
        scores = np.array([])
        canvas = build_attention_heatmap(
            registry, scores, canvas_shape=(50, 50), patch_size=256, downsample_factor=8.0,
        )
        assert (canvas == 0).all()


# ---------------------------------------------------------------------------
# colorize_heatmap
# ---------------------------------------------------------------------------

class TestColorizeHeatmap:

    def test_output_bgr_3_channels(self) -> None:
        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        bgr = colorize_heatmap(gray)
        assert bgr.shape == (50, 50, 3)
        assert bgr.dtype == np.uint8

    def test_jet_colormap_default(self) -> None:
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[5, 5] = 255
        bgr = colorize_heatmap(gray, colormap=cv2.COLORMAP_JET)
        assert bgr.shape == (10, 10, 3)


# ---------------------------------------------------------------------------
# overlay_heatmap_on_thumbnail
# ---------------------------------------------------------------------------

class TestOverlay:

    def test_output_same_size_as_thumbnail(self) -> None:
        thumb = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        heatmap_bgr = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = overlay_heatmap_on_thumbnail(thumb, heatmap_bgr)
        assert result.shape == (100, 80, 3)
        assert result.dtype == np.uint8

    def test_resize_if_different(self) -> None:
        thumb = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        heatmap_bgr = np.random.randint(0, 255, (50, 40, 3), dtype=np.uint8)
        result = overlay_heatmap_on_thumbnail(thumb, heatmap_bgr)
        assert result.shape == (100, 80, 3)


