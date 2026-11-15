"""Extraction de patchs 256x256 a partir des zones tissulaires d'une WSI.

Refactoring du script procédural de prétraitement OpenSlide :
  - Calcul explicite de scale_x / scale_y (non-isotrope)
  - Taille de fenêtre masque en W et H séparés (mask_patch_w / mask_patch_h)
  - Lecture des métadonnées physiques mpp_x / mpp_y
  - Option use_smallest_level pour brancher sur level_count - 1
  - Normalisation Macenko intégrée avant sauvegarde
  - Sortie dans data/interim/patches/ (conforme à l'arborescence du projet)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from prostate_mil.utils.io import ensure_dir

from prostate_mil.data.preprocessing.tissue_segmentation import compute_hsv_otsu_mask, refine_tissue_mask
from prostate_mil.data.wsi.wsi_reader import WSIReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Métadonnées physiques de la lame
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SlideInfo:
    """Informations physiques extraites des propriétés OpenSlide.

    Ces valeurs sont utiles pour les phases suivantes (heatmaps, MPP-aware
    feature extraction).
    """

    level_count: int
    level_dimensions: list[tuple[int, int]]
    mask_level: int
    mask_level_dimensions: tuple[int, int]
    scale_x: float   # facteur thumbnail → niveau 0 en X
    scale_y: float   # facteur thumbnail → niveau 0 en Y
    mpp_x: float | None = None   # microns par pixel en X (None si inconnu)
    mpp_y: float | None = None   # microns par pixel en Y (None si inconnu)


# ---------------------------------------------------------------------------
# Métadonnées d'un patch extrait
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PatchMetadata:
    """Métadonnées d'un patch extrait.

    Les coordonnées x / y sont exprimées en pixels niveau-0 (full resolution),
    ce qui permet de reconstruire les heatmaps de manière exacte à n'importe
    quelle magnification.
    """

    patch_id: str
    x: int                    # coordonnée gauche en pixels niveau-0
    y: int                    # coordonnée haute en pixels niveau-0
    spatial_variance: float
    mpp_x: float | None = None
    mpp_y: float | None = None


# ---------------------------------------------------------------------------
# Extracteur principal
# ---------------------------------------------------------------------------

class PatchExtractor:
    """Tesselle une WSI en ignorant le fond grâce à un masque tissu HSV-Otsu.

    Logique intégrée depuis le script de prétraitement du binôme :

    1. Ouvrir la lame avec OpenSlide via WSIReader.
    2. Lire le thumbnail au niveau mask_level (smallest ou recommandé).
    3. Calculer scale_x / scale_y séparément (pyramides non-isotropes).
    4. Construire un masque tissu via Otsu HSV + raffinement morphologique.
    5. Parcourir la grille avec des fenêtres mask_patch_w × mask_patch_h.
    6. Pour chaque fenêtre avec ratio tissu ≥ threshold :
       - Lire le patch au niveau 0 avec read_region(location=(x_l0, y_l0), …)
       - Calculer la variance spatiale.
       - Enregistrer les métadonnées (pas de sauvegarde d'image).
    """

    def __init__(
        self,
        patch_size: int = 256,
        tissue_coverage_threshold: float = 0.15,
        use_smallest_level: bool = False,
        mask_level: int | None = None,
        max_mask_dimension: int = 4096,
        min_tissue_component_area: int = 128,
    ) -> None:
        """Initialise l'extracteur.

        Args:
            patch_size: Taille des patchs extraits en pixels au niveau 0.
            tissue_coverage_threshold: Ratio minimum de tissu pour retenir
                une fenêtre candidate (identique au seuil 0.15 du binôme).
            use_smallest_level: Si True, utilise level_count - 1 comme niveau
                de masque (comportement du script du binôme). Si False, utilise
                la sélection automatique via max_mask_dimension.
            mask_level: Force un niveau pyramidal précis pour la segmentation.
                Ignoré si use_smallest_level est True.
            max_mask_dimension: Dimension cible maximale pour la sélection
                automatique du niveau de masque.
            min_tissue_component_area: Aire minimale (pixels masque) d'une
                composante connexe conservée après raffinement.
        """

        if patch_size <= 0:
            raise ValueError("`patch_size` doit être strictement positif.")
        if not 0.0 <= tissue_coverage_threshold <= 1.0:
            raise ValueError("`tissue_coverage_threshold` doit être dans [0, 1].")

        self.patch_size = patch_size
        self.tissue_coverage_threshold = tissue_coverage_threshold
        self.use_smallest_level = use_smallest_level
        self.mask_level = mask_level
        self.max_mask_dimension = max_mask_dimension
        self.min_tissue_component_area = min_tissue_component_area

    # ------------------------------------------------------------------
    # Constructeur alternatif
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "PreprocessingConfig") -> "PatchExtractor":
        """Construit un PatchExtractor depuis un PreprocessingConfig."""

        return cls(
            patch_size=config.patch_size,
            tissue_coverage_threshold=config.tissue_threshold,
            mask_level=config.mask_level,
            max_mask_dimension=config.max_mask_dimension,
            min_tissue_component_area=config.min_tissue_component_area,
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def read_slide_info(self, reader: WSIReader) -> SlideInfo:
        """Extrait les paramètres physiques et géométriques de la lame.

        Reprend la logique du binôme :
          - level_count / level_dimensions
          - scale_x  = W_level0 / W_mask_level  (facteur thumbnail→niveau 0)
          - scale_y  = H_level0 / H_mask_level
          - mpp_x / mpp_y depuis les propriétés OpenSlide
        """

        level_count = reader.get_num_levels()
        level_dimensions = [
            reader.get_level_dimensions(lvl) for lvl in range(level_count)
        ]

        mask_level = self._resolve_mask_level(reader)
        mask_w, mask_h = reader.get_level_dimensions(mask_level)
        full_w, full_h = reader.get_dimensions()  # niveau 0

        # Calcul explicite des facteurs d'échelle séparés (binôme)
        scale_x = full_w / max(mask_w, 1)
        scale_y = full_h / max(mask_h, 1)

        # Métadonnées physiques optionnelles
        mpp_x, mpp_y = reader.get_mpp()

        if mpp_x is None:
            logger.warning(
                "mpp_x introuvable dans les propriétés OpenSlide. "
                "La magnification physique est inconnue pour cette lame."
            )

        return SlideInfo(
            level_count=level_count,
            level_dimensions=level_dimensions,
            mask_level=mask_level,
            mask_level_dimensions=(mask_w, mask_h),
            scale_x=scale_x,
            scale_y=scale_y,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
        )

    def iter_patch_coordinates(
        self,
        tissue_mask: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> Iterator[tuple[int, int]]:
        """Génère les coordonnées niveau-0 des patchs admissibles.

        Calcul repris du binôme :
            mask_patch_w = PATCH_SIZE / scale_x   → taille d'une fenêtre en X
            mask_patch_h = PATCH_SIZE / scale_y   → taille d'une fenêtre en Y
            x_level0 = col * scale_x
            y_level0 = row * scale_y

        Les axes X et Y sont traités indépendamment pour gérer correctement
        les pyramides non-isotropes.

        Args:
            tissue_mask: Masque binaire 2D (H, W), dtype uint8.
            scale_x: Facteur thumbnail→niveau 0 en X (col de masque → col lame).
            scale_y: Facteur thumbnail→niveau 0 en Y (row de masque → row lame).

        Yields:
            (x_level0, y_level0) : coin supérieur gauche du patch en
            coordonnées niveau-0, prêt à passer dans read_region().
        """

        if tissue_mask.ndim != 2:
            raise ValueError("`tissue_mask` doit être une image 2D [H, W].")
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("`scale_x` et `scale_y` doivent être > 0.")

        # Taille d'un patch exprimée dans l'espace du masque (X et Y séparés)
        mask_patch_w = max(1, int(round(self.patch_size / scale_x)))
        mask_patch_h = max(1, int(round(self.patch_size / scale_y)))

        mask_height, mask_width = tissue_mask.shape

        if mask_patch_w > mask_width or mask_patch_h > mask_height:
            logger.warning(
                "La fenêtre masque (%dx%d) dépasse les dimensions du masque "
                "(%dx%d). Aucun patch généré.",
                mask_patch_w, mask_patch_h, mask_width, mask_height,
            )
            return

        for row in range(0, mask_height - mask_patch_h + 1, mask_patch_h):
            for col in range(0, mask_width - mask_patch_w + 1, mask_patch_w):

                # Évaluer la couverture tissulaire dans la fenêtre courante
                window = tissue_mask[
                    row : row + mask_patch_h,
                    col : col + mask_patch_w,
                ]
                tissue_ratio = float((window > 0).mean())

                if tissue_ratio < self.tissue_coverage_threshold:
                    continue  # fenêtre de fond → ignorer

                # Remonter en coordonnées niveau-0 (logique du binôme)
                x_level0 = int(round(col * scale_x))
                y_level0 = int(round(row * scale_y))
                yield x_level0, y_level0

    def compute_spatial_variance(self, patch_rgb: np.ndarray) -> float:
        """Calcule la variance spatiale du patch via le Laplacien.

        La variance du Laplacien mesure les transitions haute frequence
        (contours, texture). Un patch flou ou uniforme aura une valeur
        basse ; un patch riche en tissu aura une valeur elevee.
        """

        gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def extract_and_save_patches(
        self,
        slide_path: str | Path,
        output_dir: str | Path = Path("data/interim"),
        show_progress: bool = True,
    ) -> tuple[list[PatchMetadata], SlideInfo]:
        """Pipeline principal de patchification pour une lame WSI.

        Sorties dans output_dir/<slide_name>/ :
          - thumbnail_level_N.png  → référence visuelle du niveau masque
          - tissue_mask_level_N.png → masque tissu binaire

        Args:
            slide_path: Chemin vers le fichier WSI (.svs, .tiff, .ndpi, …).
            output_dir: Racine de sortie. Par défaut data/interim/ conforme
                à l'arborescence du projet.

        Returns:
            (records, slide_info) : liste de PatchMetadata + SlideInfo.
        """

        slide_path = Path(slide_path)
        output_dir = ensure_dir(Path(output_dir) / slide_path.stem)
        slide_name = slide_path.stem

        records: list[PatchMetadata] = []

        try:
            with WSIReader(slide_path) as reader:

                # ── Informations physiques de la lame (logique du binôme) ──
                info = self.read_slide_info(reader)

                logger.info(
                    "%s — %d niveaux | thumbnail: %dx%d | "
                    "scale_x=%.2f scale_y=%.2f | mpp_x=%s",
                    slide_name,
                    info.level_count,
                    info.mask_level_dimensions[0],
                    info.mask_level_dimensions[1],
                    info.scale_x,
                    info.scale_y,
                    info.mpp_x,
                )

                # ── Segmentation tissu ────────────────────────────────────
                level_thumbnail = reader.read_level_rgb(info.mask_level)
                tissue_mask = refine_tissue_mask(
                    compute_hsv_otsu_mask(level_thumbnail),
                    min_area=self.min_tissue_component_area,
                )
                self._save_quality_control_images(
                    output_dir=output_dir,
                    thumbnail_rgb=level_thumbnail,
                    tissue_mask=tissue_mask,
                    mask_level=info.mask_level,
                )

                # ── Boucle d'extraction ──────────────────────────────────
                coordinates = list(self.iter_patch_coordinates(
                    tissue_mask=tissue_mask,
                    scale_x=info.scale_x,
                    scale_y=info.scale_y,
                ))
                logger.info(
                    "%s — %d patchs candidats identifies", slide_name, len(coordinates),
                )

                pbar = tqdm(
                    coordinates, 
                    desc=f"Extracting {slide_name}", 
                    unit="patch",
                    disable=not show_progress
                )
                for index, (x, y) in enumerate(pbar, start=1):

                    # Lecture du patch au niveau 0 (full resolution)
                    # location = (x, y) en coordonnées niveau-0
                    patch_rgb = reader.read_region_rgb(
                        location=(x, y),
                        level=0,
                        size=(self.patch_size, self.patch_size),
                    )

                    spatial_variance = self.compute_spatial_variance(patch_rgb)
                    patch_id = f"{slide_name}_patch_{index:06d}_x{x:06d}_y{y:06d}"

                    records.append(
                        PatchMetadata(
                            patch_id=patch_id,
                            x=x,
                            y=y,
                            spatial_variance=spatial_variance,
                            mpp_x=info.mpp_x,
                            mpp_y=info.mpp_y,
                        )
                    )

        finally:
            pass

        logger.info(
            "%s — %d patchs extraits → %s",
            slide_name, len(records), output_dir,
        )
        return records, info

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _resolve_mask_level(self, reader: WSIReader) -> int:
        """Détermine le niveau pyramidal utilisé pour la segmentation.

        Si use_smallest_level est True, retourne level_count - 1 (logique du
        binôme). Sinon, utilise la sélection automatique par max_mask_dimension,
        ou force mask_level si fourni.
        """

        if self.use_smallest_level:
            return reader.get_num_levels() - 1

        if self.mask_level is not None:
            return self.mask_level

        return reader.get_recommended_mask_level(
            max_dimension=self.max_mask_dimension
        )

    def _save_quality_control_images(
        self,
        output_dir: Path,
        thumbnail_rgb: np.ndarray,
        tissue_mask: np.ndarray,
        mask_level: int,
    ) -> None:
        """Sauvegarde les artefacts de contrôle qualité."""

        Image.fromarray(thumbnail_rgb).save(
            output_dir / f"thumbnail_level_{mask_level}.png"
        )
        Image.fromarray(tissue_mask).save(
            output_dir / f"tissue_mask_level_{mask_level}.png"
        )

