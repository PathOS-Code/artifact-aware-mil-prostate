"""Wrappers de lecture OpenSlide pour Whole Slide Images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openslide
from PIL import Image


class WSIReader:
    """Encapsule les operations de base sur une WSI.

    Cette classe sert de point d'entree unique pour ouvrir une lame,
    recuperer ses dimensions, lire des regions et construire un thumbnail.
    """

    def __init__(self, slide_path: str | Path) -> None:
        self.slide_path = Path(slide_path)
        self.slide: openslide.OpenSlide | None = None

    def __enter__(self) -> "WSIReader":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def open(self) -> None:
        """Ouvre la lame avec OpenSlide."""

        if not self.slide_path.exists():
            raise FileNotFoundError(f"WSI introuvable: {self.slide_path}")
        self.slide = openslide.OpenSlide(str(self.slide_path))

    def close(self) -> None:
        """Ferme le handle OpenSlide si necessaire."""

        if self.slide is not None:
            self.slide.close()
            self.slide = None

    def get_dimensions(self) -> tuple[int, int]:
        """Retourne les dimensions de niveau 0 de la lame."""

        return tuple(int(value) for value in self._require_open().dimensions)

    def get_num_levels(self) -> int:
        """Retourne le nombre de niveaux pyramidaux disponibles."""

        return int(self._require_open().level_count)

    def get_level_dimensions(self, level: int) -> tuple[int, int]:
        """Retourne les dimensions d'un niveau pyramidal donne."""

        slide = self._require_open()
        self._validate_level(level)
        return tuple(int(value) for value in slide.level_dimensions[level])

    def get_level_downsample(self, level: int) -> float:
        """Retourne le facteur de downsample d'un niveau pyramidal."""

        slide = self._require_open()
        self._validate_level(level)
        return float(slide.level_downsamples[level])

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Selectionne le meilleur niveau OpenSlide pour un downsample cible."""

        slide = self._require_open()
        return int(slide.get_best_level_for_downsample(max(1.0, downsample)))

    def get_recommended_mask_level(self, max_dimension: int = 4096) -> int:
        """Choisit un niveau de lecture compact pour la segmentation tissulaire."""

        width, height = self.get_dimensions()
        target_downsample = max(width, height) / max(1, max_dimension)
        return self.get_best_level_for_downsample(max(1.0, target_downsample))

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        """Lit une region de la lame.

        Args:
            location: Coordonnees absolues sur le niveau 0.
            level: Niveau pyramidal OpenSlide.
            size: Taille de la fenetre a lire.
        """

        slide = self._require_open()
        self._validate_level(level)
        return slide.read_region(location, level, size)

    def read_region_rgb(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """Lit une region et retourne un tableau RGB `uint8`."""

        region = self.read_region(location=location, level=level, size=size)
        return np.asarray(region.convert("RGB"), dtype=np.uint8)

    def read_level_rgb(self, level: int) -> np.ndarray:
        """Charge entierement un niveau pyramidal sous forme RGB.

        Ce helper est reserve aux niveaux reduits utilises pour segmentation.
        """

        width, height = self.get_level_dimensions(level)
        return self.read_region_rgb(location=(0, 0), level=level, size=(width, height))

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        """Construit un thumbnail RGB de la lame pour segmentation ou overlay."""

        slide = self._require_open()
        thumbnail = slide.get_thumbnail(size).convert("RGB")
        return np.asarray(thumbnail, dtype=np.uint8)

    def get_mpp(self) -> tuple[float | None, float | None]:
        """Retourne (mpp_x, mpp_y) depuis les proprietes OpenSlide."""

        slide = self._require_open()
        raw_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        raw_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        return (
            float(raw_x) if raw_x is not None else None,
            float(raw_y) if raw_y is not None else None,
        )

    def get_properties(self) -> dict[str, str]:
        """Retourne l'ensemble des proprietes OpenSlide de la lame."""

        return dict(self._require_open().properties)

    def _require_open(self) -> openslide.OpenSlide:
        """Retourne le handle OpenSlide ou leve une erreur explicite."""

        if self.slide is None:
            raise RuntimeError("La WSI doit etre ouverte avant utilisation.")
        return self.slide

    def _validate_level(self, level: int) -> None:
        """Valide qu'un niveau pyramidal existe dans la lame."""

        if not 0 <= level < self.get_num_levels():
            raise ValueError(
                f"Le niveau {level} est invalide pour une lame avec "
                f"{self.get_num_levels()} niveaux."
            )
