"""Creation et sauvegarde du registre CSV de patchs.

Le registre inclut les champs physiques (mpp_x/mpp_y) et la trace de
normalisation (is_normalized) ajoutés lors du refactoring Phase 1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from prostate_mil.data.patches.patch_extraction import PatchMetadata


def build_patch_registry(records: list[PatchMetadata]) -> pd.DataFrame:
    """Convertit les metadonnees de patchs en DataFrame tabulaire."""

    return pd.DataFrame(
        [
            {
                "patch_id": record.patch_id,
                "coord_x": record.x,
                "coord_y": record.y,
                "variance_spatiale": record.spatial_variance,
                "mpp_x": record.mpp_x,
                "mpp_y": record.mpp_y,
            }
            for record in records
        ]
    )


def save_patch_registry(records: list[PatchMetadata], csv_path: str | Path) -> Path:
    """Sauvegarde le fichier de registre CSV attendu par les phases suivantes."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    build_patch_registry(records).to_csv(csv_path, index=False)
    return csv_path
