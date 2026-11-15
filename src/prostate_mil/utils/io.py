"""Fonctions utilitaires pour charger/sauvegarder CSV, tenseurs et chemins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    """Cree le dossier si necessaire et retourne le chemin normalise."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_patch_registry(csv_path: str | Path) -> pd.DataFrame:
    """Charge le CSV de patchs genere pendant la phase 1."""

    return pd.read_csv(csv_path)


def save_tensor(tensor: torch.Tensor, output_path: str | Path) -> Path:
    """Sauvegarde un tenseur PyTorch au format `.pt`."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)
    return output_path
