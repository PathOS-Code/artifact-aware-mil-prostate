"""Tests unitaires — extraction de patchs, variance spatiale et registre CSV.

Ces tests utilisent des masques et patchs synthetiques, sans dependance
a OpenSlide ni a des fichiers WSI reels.

Execution :
    python -m pytest tests/test_patch_extraction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prostate_mil.data.patches.patch_extraction import PatchExtractor, PatchMetadata
from prostate_mil.data.datasets.registry import build_patch_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor() -> PatchExtractor:
    """PatchExtractor avec parametres par defaut."""
    return PatchExtractor(patch_size=256, tissue_coverage_threshold=0.5)


# ---------------------------------------------------------------------------
# iter_patch_coordinates
# ---------------------------------------------------------------------------

class TestIterPatchCoordinates:
    """Tests pour la generation de coordonnees de patchs."""

    def test_full_tissue_mask(self, extractor: PatchExtractor) -> None:
        """Un masque 100% tissu doit produire une grille complete."""
        scale_x, scale_y = 4.0, 4.0
        mask_w, mask_h = 256, 128
        mask = np.ones((mask_h, mask_w), dtype=np.uint8) * 255

        # Taille d'une fenetre dans l'espace masque
        mask_patch_w = round(256 / scale_x)  # 64
        mask_patch_h = round(256 / scale_y)  # 64

        expected_cols = mask_w // mask_patch_w  # 4
        expected_rows = mask_h // mask_patch_h  # 2
        expected_count = expected_cols * expected_rows  # 8

        coords = list(extractor.iter_patch_coordinates(mask, scale_x, scale_y))
        assert len(coords) == expected_count

        # Les coordonnees sont en niveau-0 : espacement = patch_size = 256
        xs = sorted(set(c[0] for c in coords))
        ys = sorted(set(c[1] for c in coords))
        assert xs == [round(col * scale_x) for col in range(0, mask_w, mask_patch_w)]
        assert ys == [round(row * scale_y) for row in range(0, mask_h, mask_patch_h)]

    def test_empty_mask(self, extractor: PatchExtractor) -> None:
        """Un masque vide ne doit produire aucun patch."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        coords = list(extractor.iter_patch_coordinates(mask, 2.0, 2.0))
        assert len(coords) == 0

    def test_partial_tissue(self, extractor: PatchExtractor) -> None:
        """Seules les fenetres avec couverture >= threshold sont retenues."""
        scale_x, scale_y = 2.0, 2.0
        mask_patch_w = round(256 / scale_x)  # 128
        mask_patch_h = round(256 / scale_y)  # 128

        # Masque 256x256 : 2x2 fenetres dans l'espace masque
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Remplir uniquement le quadrant haut-gauche (100% tissu)
        mask[:mask_patch_h, :mask_patch_w] = 255

        coords = list(extractor.iter_patch_coordinates(mask, scale_x, scale_y))
        assert len(coords) == 1
        assert coords[0] == (0, 0)

    def test_invalid_mask_raises(self, extractor: PatchExtractor) -> None:
        """Un masque 3D doit lever une ValueError."""
        mask_3d = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D"):
            list(extractor.iter_patch_coordinates(mask_3d, 2.0, 2.0))

    def test_invalid_scale_raises(self, extractor: PatchExtractor) -> None:
        """Un scale negatif doit lever une ValueError."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        with pytest.raises(ValueError, match="scale"):
            list(extractor.iter_patch_coordinates(mask, -1.0, 2.0))


# ---------------------------------------------------------------------------
# compute_spatial_variance
# ---------------------------------------------------------------------------

class TestComputeSpatialVariance:
    """Tests pour la metrique de variance spatiale (Laplacien)."""

    def test_uniform_patch_low_variance(self, extractor: PatchExtractor) -> None:
        """Un patch uniforme doit avoir une variance proche de zero."""
        uniform = np.full((256, 256, 3), 128, dtype=np.uint8)
        var = extractor.compute_spatial_variance(uniform)
        assert var < 1.0

    def test_textured_patch_high_variance(self, extractor: PatchExtractor) -> None:
        """Un patch avec des contours nets doit avoir une variance elevee."""
        # Damier de 8x8 blocs : beaucoup de transitions haute frequence
        checker = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                if (i // 32 + j // 32) % 2 == 0:
                    checker[i, j] = 255
        var = extractor.compute_spatial_variance(checker)
        assert var > 100.0

    def test_textured_greater_than_uniform(self, extractor: PatchExtractor) -> None:
        """La variance d'un patch texture doit etre superieure a un patch uniforme."""
        uniform = np.full((256, 256, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(42)
        noisy = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

        var_uniform = extractor.compute_spatial_variance(uniform)
        var_noisy = extractor.compute_spatial_variance(noisy)
        assert var_noisy > var_uniform


# ---------------------------------------------------------------------------
# build_patch_registry
# ---------------------------------------------------------------------------

class TestBuildPatchRegistry:
    """Tests pour la creation du registre CSV."""

    def test_registry_columns(self) -> None:
        """Le DataFrame doit contenir toutes les colonnes attendues."""
        records = [
            PatchMetadata(
                patch_id="slide_patch_000001_x000000_y000000",
                x=0, y=0,
                spatial_variance=42.5,
                mpp_x=0.25,
                mpp_y=0.25,
            ),
        ]
        df = build_patch_registry(records)

        expected_cols = {
            "patch_id", "coord_x", "coord_y", "variance_spatiale", "mpp_x", "mpp_y",
        }
        assert set(df.columns) == expected_cols

    def test_registry_values(self) -> None:
        """Les valeurs du DataFrame doivent correspondre aux PatchMetadata."""
        records = [
            PatchMetadata(
                patch_id="s1_p1", x=100, y=200,
                spatial_variance=33.3,
                mpp_x=0.5, mpp_y=0.5,
            ),
            PatchMetadata(
                patch_id="s1_p2", x=356, y=400,
                spatial_variance=77.7,
                mpp_x=0.5, mpp_y=0.5,
            ),
        ]
        df = build_patch_registry(records)

        assert len(df) == 2
        assert df.iloc[0]["coord_x"] == 100
        assert df.iloc[1]["variance_spatiale"] == 77.7

    def test_empty_registry(self) -> None:
        """Un registre vide doit produire un DataFrame vide."""
        df = build_patch_registry([])
        assert len(df) == 0
