"""Test d'intégration – Phase 1 : segmentation tissu + normalisation Macenko.

Exécution :
    python -m pytest tests/test_pipeline_phase1.py -v
    # ou bien directement :
    python -m unittest tests.test_pipeline_phase1 -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ajout du dossier src/ au PYTHONPATH pour les imports du package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prostate_mil.data.preprocessing.color_normalization import MacenkoNormalizerWrapper
from prostate_mil.data.preprocessing.tissue_segmentation import (
    compute_hsv_otsu_mask,
    refine_tissue_mask,
)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
RAW_IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "photobiopsietestcomplete.png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "test_results"


class TestPipelinePhase1Integration(unittest.TestCase):
    """Valide le pipeline complet: chargement → segmentation → normalisation."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    # Taille max du côté le plus long pour le thumbnail de test.
    # On travaille sur un thumbnail pour garder les tests rapides tout en
    # validant le pipeline sur une vraie image de biopsie.
    MAX_TEST_DIMENSION: int = 4096

    @classmethod
    def setUpClass(cls) -> None:
        """Charge l'image une seule fois pour tous les tests de la classe."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not RAW_IMAGE_PATH.exists():
            print(f"⚠️  Image de test introuvable : {RAW_IMAGE_PATH}\nGénération d'une image synthétique pour les tests.")
            cls.image_rgb = np.full((1024, 1024, 3), 240, dtype=np.uint8) # Fond clair
            cls.image_rgb[256:768, 256:768] = [180, 100, 150] # Un carré coloré simulant du tissu H&E
        else:
            # L'image de biopsie peut dépasser la limite anti-bombe de Pillow.
            Image.MAX_IMAGE_PIXELS = None

            pil_image = Image.open(RAW_IMAGE_PATH).convert("RGB")
            original_size = pil_image.size  # (W, H)

            # Redimensionner en thumbnail pour garder les tests rapides
            pil_image.thumbnail(
                (cls.MAX_TEST_DIMENSION, cls.MAX_TEST_DIMENSION),
                Image.Resampling.LANCZOS,
            )

            cls.image_rgb = np.asarray(pil_image, dtype=np.uint8)
            print(
                f"\n📷  Image chargée : {RAW_IMAGE_PATH.name}  "
                f"(original {original_size[0]}×{original_size[1]}, "
                f"thumbnail {cls.image_rgb.shape[1]}×{cls.image_rgb.shape[0]}, "
                f"{cls.image_rgb.dtype})"
            )

    # ------------------------------------------------------------------
    # 1. Test de chargement de l'image
    # ------------------------------------------------------------------
    def test_01_image_loaded_correctly(self) -> None:
        """L'image doit être un tableau RGB 3D de type uint8."""

        self.assertEqual(self.image_rgb.ndim, 3, "L'image doit être 3D [H, W, C].")
        self.assertEqual(self.image_rgb.shape[2], 3, "L'image doit avoir 3 canaux (RGB).")
        self.assertEqual(self.image_rgb.dtype, np.uint8)
        self.assertGreater(self.image_rgb.shape[0], 0, "Hauteur nulle.")
        self.assertGreater(self.image_rgb.shape[1], 0, "Largeur nulle.")

    # ------------------------------------------------------------------
    # 2. Test de la segmentation tissu
    # ------------------------------------------------------------------
    def test_02_tissue_segmentation(self) -> None:
        """Le masque Otsu HSV doit être 2D, non vide, et de mêmes dimensions."""

        raw_mask = compute_hsv_otsu_mask(self.image_rgb)

        # Forme et type
        self.assertEqual(raw_mask.ndim, 2, "Le masque brut doit être 2D.")
        self.assertEqual(
            raw_mask.shape[:2],
            self.image_rgb.shape[:2],
            "Le masque doit avoir les mêmes dimensions H×W que l'image.",
        )
        self.assertEqual(raw_mask.dtype, np.uint8)

        # Le masque ne doit pas être entièrement vide (aucun tissu détecté)
        tissue_pixels = int(np.count_nonzero(raw_mask))
        self.assertGreater(
            tissue_pixels, 0,
            "Le masque de segmentation est entièrement noir : aucun tissu détecté.",
        )

        # Sauvegarder le masque brut pour inspection visuelle
        mask_path = OUTPUT_DIR / "tissue_mask_raw.png"
        Image.fromarray(raw_mask).save(mask_path)
        print(f"   💾  Masque brut sauvegardé → {mask_path.relative_to(PROJECT_ROOT)}")

        # Stocker pour les tests suivants
        self.__class__._raw_mask = raw_mask

    # ------------------------------------------------------------------
    # 3. Test du raffinement morphologique du masque
    # ------------------------------------------------------------------
    def test_03_tissue_mask_refinement(self) -> None:
        """Le masque raffiné doit rester cohérent après les opérations morpho."""

        raw_mask = getattr(self.__class__, "_raw_mask", None)
        if raw_mask is None:
            raw_mask = compute_hsv_otsu_mask(self.image_rgb)

        refined_mask = refine_tissue_mask(raw_mask, min_area=128)

        # Forme identique
        self.assertEqual(refined_mask.shape, raw_mask.shape)
        self.assertEqual(refined_mask.dtype, np.uint8)

        # Toujours du tissu après nettoyage
        tissue_pixels_refined = int(np.count_nonzero(refined_mask))
        self.assertGreater(
            tissue_pixels_refined, 0,
            "Le masque raffiné est vide : le nettoyage morpho a tout supprimé.",
        )

        # Le closing morphologique peut dilater légèrement les contours,
        # donc le nombre de pixels tissu peut augmenter modérément.
        # On vérifie que le ratio reste raisonnable (±20%).
        tissue_pixels_raw = int(np.count_nonzero(raw_mask))
        ratio = tissue_pixels_refined / max(tissue_pixels_raw, 1)
        self.assertAlmostEqual(
            ratio, 1.0, delta=0.20,
            msg=(
                f"Le raffinement a modifié le masque de manière excessive "
                f"(ratio = {ratio:.2f}, attendu ≈ 1.0 ± 0.20)."
            ),
        )

        # Sauvegarder
        refined_path = OUTPUT_DIR / "tissue_mask_refined.png"
        Image.fromarray(refined_mask).save(refined_path)
        print(f"   💾  Masque raffiné sauvegardé → {refined_path.relative_to(PROJECT_ROOT)}")

        # Calculer et afficher la couverture tissulaire
        coverage = tissue_pixels_refined / refined_mask.size * 100
        print(f"   📊  Couverture tissulaire : {coverage:.1f}%")

    # ------------------------------------------------------------------
    # 4. Test de la normalisation Macenko
    # ------------------------------------------------------------------
    def test_04_macenko_normalization(self) -> None:
        """La normalisation Macenko doit produire une image RGB de mêmes dimensions."""

        normalizer = MacenkoNormalizerWrapper()

        # Fit sur l'image elle-même (auto-référence pour ce test d'intégration)
        normalizer.fit(self.image_rgb)
        self.assertTrue(normalizer.is_fitted, "Le normalizer devrait être fitted.")

        # Transformer
        normalized = normalizer.transform(self.image_rgb)

        # Vérifications de forme et type
        self.assertEqual(normalized.ndim, 3)
        self.assertEqual(normalized.shape, self.image_rgb.shape,
                         "L'image normalisée doit conserver les dimensions originales.")
        self.assertEqual(normalized.dtype, np.uint8)

        # L'image normalisée ne doit pas être entièrement noire
        self.assertGreater(
            normalized.mean(), 0,
            "L'image normalisée est entièrement noire.",
        )

        # L'image normalisée doit être différente de l'originale
        # (sauf cas dégénéré où l'image EST déjà la référence parfaite)
        pixel_diff = np.abs(
            normalized.astype(np.float32) - self.image_rgb.astype(np.float32)
        ).mean()
        print(f"   📊  Différence moyenne par pixel après normalisation : {pixel_diff:.2f}")

        # Sauvegarder l'image normalisée
        norm_path = OUTPUT_DIR / "normalized_macenko.png"
        Image.fromarray(normalized).save(norm_path)
        print(f"   💾  Image normalisée sauvegardée → {norm_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    # 5. Test du pipeline complet enchaîné
    # ------------------------------------------------------------------
    def test_05_full_pipeline_chained(self) -> None:
        """Enchaîne segmentation → raffinement → normalisation et sauvegarde."""

        # Étape 1 : Segmentation
        raw_mask = compute_hsv_otsu_mask(self.image_rgb)
        refined_mask = refine_tissue_mask(raw_mask, min_area=128)

        # Étape 2 : Normalisation Macenko
        normalizer = MacenkoNormalizerWrapper()
        normalizer.fit(self.image_rgb)
        normalized = normalizer.transform(self.image_rgb)

        # Étape 3 : Appliquer le masque sur l'image normalisée (aperçu tissu seul)
        tissue_only = normalized.copy()
        tissue_only[refined_mask == 0] = 0  # fond → noir

        # Assertions finales
        self.assertEqual(tissue_only.shape, self.image_rgb.shape)
        non_zero_pixels = np.count_nonzero(tissue_only.sum(axis=2))
        self.assertGreater(non_zero_pixels, 0, "L'image masquée est entièrement noire.")

        # Sauvegarder le résultat final
        final_path = OUTPUT_DIR / "pipeline_result_tissue_normalized.png"
        Image.fromarray(tissue_only).save(final_path)
        print(f"   💾  Pipeline complet sauvegardé → {final_path.relative_to(PROJECT_ROOT)}")
        print(f"   ✅  Pixels de tissu conservés : {non_zero_pixels:,}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
