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


def find_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the run's weight file: best_model.safetensors, else best_model.pt."""

    run_dir = Path(run_dir)
    for name in ("best_model.safetensors", "best_model.pt"):
        if (run_dir / name).exists():
            return run_dir / name
    return None


def load_model_state(run_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load a run's model state_dict on CPU.

    Released weights are ``best_model.safetensors``; training writes
    ``best_model.pt`` (state_dict plus optimizer state and metadata).
    """

    path = find_checkpoint(run_dir)
    if path is None:
        raise FileNotFoundError(f"No best_model.safetensors or best_model.pt in {run_dir}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
