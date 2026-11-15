"""Tests for loading run weights from best_model.pt or best_model.safetensors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prostate_mil.utils.io import find_checkpoint, load_model_state


def _state() -> dict[str, torch.Tensor]:
    return {"layer.weight": torch.randn(4, 3), "layer.bias": torch.randn(4)}


def test_loads_training_checkpoint(tmp_path: Path) -> None:
    state = _state()
    torch.save({"model_state_dict": state, "optimizer_state_dict": {}}, tmp_path / "best_model.pt")
    loaded = load_model_state(tmp_path)
    assert all(torch.equal(loaded[k], state[k]) for k in state)


def test_prefers_safetensors(tmp_path: Path) -> None:
    released = _state()
    save_file(released, str(tmp_path / "best_model.safetensors"))
    torch.save({"model_state_dict": _state()}, tmp_path / "best_model.pt")
    assert find_checkpoint(tmp_path).name == "best_model.safetensors"
    loaded = load_model_state(tmp_path)
    assert all(torch.equal(loaded[k], released[k]) for k in released)


def test_missing_weights_raise(tmp_path: Path) -> None:
    assert find_checkpoint(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        load_model_state(tmp_path)
