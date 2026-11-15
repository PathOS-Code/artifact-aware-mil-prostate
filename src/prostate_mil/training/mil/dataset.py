"""Dataset PyTorch pour entrainement MIL a partir d'embeddings pre-calcules."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SlideEmbeddingDataset(Dataset):
    """Charge les embeddings `.pt` de Phase 2 avec les labels ISUP.

    Chaque element retourne un bag complet (1 slide = 1 sample) avec
    un nombre variable de patches N. Utiliser avec batch_size=1.
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        labels_csv: str | Path,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        labels_df = pd.read_csv(labels_csv)

        # Déterminer la colonne du nom de la slide (PANDA: image_id, Projets: slide_name)
        id_col = "image_id" if "image_id" in labels_df.columns else "slide_name"
        label_col = "isup_grade" # Commun aux deux formats

        # Filtrer les slides qui ont un fichier .pt correspondant
        self.samples: list[tuple[Path, int]] = []
        for _, row in labels_df.iterrows():
            slide_name = row[id_col]
            pt_path = self.embeddings_dir / f"{slide_name}.pt"
            if pt_path.exists():
                self.samples.append((pt_path, int(row[label_col])))
            else:
                logger.warning("Embeddings manquants pour %s", slide_name)

        if not self.samples:
            raise ValueError(
                f"Aucun fichier .pt trouve dans {self.embeddings_dir} "
                f"correspondant aux labels de {labels_csv}."
            )
        logger.info("%d slides chargees depuis %s", len(self.samples), labels_csv)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | int | str]:
        pt_path, label = self.samples[index]
        data = torch.load(pt_path, weights_only=False)

        return {
            "embeddings": data["embeddings"],            # [N, 768]
            "spatial_variances": data["spatial_variances"],  # [N]
            "label": label,
            "slide_name": pt_path.stem,
        }
