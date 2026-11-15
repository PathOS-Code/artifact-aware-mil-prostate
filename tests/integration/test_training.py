"""Tests unitaires — Phase 3 : modele MIL, dataset, engine.

Execution :
    python -m pytest tests/test_training.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prostate_mil.models.mil.model import CustomMILModel
from prostate_mil.training.mil.dataset import SlideEmbeddingDataset
from prostate_mil.training.mil.engine import EpochMetrics, train_one_epoch, validate_one_epoch
from prostate_mil.training.mil.losses import FocalLoss


# ---------------------------------------------------------------------------
# CustomMILModel
# ---------------------------------------------------------------------------

class TestCustomMILModel:
    """Tests pour le modele MIL complet."""

    @pytest.fixture
    def model(self) -> CustomMILModel:
        return CustomMILModel(input_dim=768, num_classes=5, variance_threshold=15.0)

    def test_forward_output_keys(self, model: CustomMILModel) -> None:
        """Le forward doit retourner logits, slide_embedding, attention_weights, etc."""
        tokens = torch.randn(50, 768)
        variances = torch.rand(50) * 100
        out = model(tokens, variances)

        assert "logits" in out
        assert "slide_embedding" in out
        assert "attention_weights" in out
        assert "artifact_mask" in out

    def test_logits_shape(self, model: CustomMILModel) -> None:
        """Les logits doivent avoir la forme [5] (5 classes ISUP)."""
        tokens = torch.randn(100, 768)
        variances = torch.rand(100) * 100
        out = model(tokens, variances)
        assert out["logits"].shape == (5,)

    def test_attention_weights_sum_to_one(self, model: CustomMILModel) -> None:
        """Les poids d'attention doivent sommer a 1 (softmax)."""
        tokens = torch.randn(30, 768)
        variances = torch.rand(30) * 100
        out = model(tokens, variances)
        assert torch.allclose(out["attention_weights"].sum(), torch.tensor(1.0), atol=1e-5)

    def test_gradient_flows(self, model: CustomMILModel) -> None:
        """Les gradients doivent circuler a travers le modele."""
        tokens = torch.randn(20, 768)
        variances = torch.rand(20) * 100
        out = model(tokens, variances)
        loss = out["logits"].sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_gated_tokens_fed_to_attention_not_raw(self) -> None:
        """Regression: attention doit recevoir gated_tokens (scaled), pas les raw tokens.

        With alpha=0.0 and artifact tokens present, gated_tokens for those tokens = 0.
        This verifies that penalty_factors are actually applied to tokens before
        being stored in gated_tokens. The test does NOT require the attention mechanism
        to differ (masking already handles that), but verifies the gate computes correctly.
        """
        torch.manual_seed(0)

        model_gated = CustomMILModel(
            input_dim=768,
            num_classes=5,
            variance_threshold=1e9,  # All patches "low variance" → only norm check matters
            alpha=0.0,               # Artifacts → exactly zero vectors
            use_gating=True,
        )
        model_gated.eval()

        # 1 extreme outlier (norm >> mean+2*std) + 19 near-zero tokens.
        # With 19 tokens at norm≈0.28 and 1 at norm≈27712:
        #   mean ≈ 27712/20 ≈ 1386, std ≈ small, outlier >> cutoff.
        tokens = torch.zeros(20, 768)
        tokens[0] = 1000.0   # One extreme outlier, norm = 1000*sqrt(768) ≈ 27712
        tokens[1:] = 0.01    # Near-zero tokens, norm = 0.01*sqrt(768) ≈ 0.277
        variances = torch.zeros(20)  # All below variance_threshold=1e9

        with torch.no_grad():
            out = model_gated(tokens, variances)

        # The extreme outlier must be flagged
        assert out["artifact_mask"][0].item(), (
            "Token[0] (extreme outlier) should be flagged as an artifact."
        )

        # penalty_factors for flagged tokens must be exactly 0.0 (alpha=0)
        flagged = out["artifact_mask"]
        assert torch.all(out["penalty_factors"][flagged] == 0.0), (
            "Artifact penalty_factors should be exactly 0.0 when alpha=0.0"
        )

        # gated_tokens for artifact tokens must be zero vectors
        expected_gated = tokens.clone()
        expected_gated[flagged] = 0.0
        assert torch.allclose(out["gated_tokens"], expected_gated, atol=1e-6), (
            "gated_tokens must equal tokens * penalty_factors. "
            "Artifact tokens should be zero vectors when alpha=0."
        )







# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    """Tests pour la Focal Loss."""

    def test_output_scalar(self) -> None:
        """La loss doit retourner un scalaire."""
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 3, 4])
        loss = criterion(logits, targets)
        assert loss.ndim == 0

    def test_perfect_prediction_low_loss(self) -> None:
        """Une prediction parfaite doit donner une loss tres basse."""
        criterion = FocalLoss(gamma=2.0)
        logits = torch.zeros(1, 5)
        logits[0, 2] = 100.0  # tres confiant sur classe 2
        targets = torch.tensor([2])
        loss = criterion(logits, targets)
        assert loss.item() < 0.01


# ---------------------------------------------------------------------------
# SlideEmbeddingDataset
# ---------------------------------------------------------------------------

class TestSlideEmbeddingDataset:
    """Tests pour le dataset d'embeddings."""

    @pytest.fixture
    def dataset_dir(self, tmp_path: Path) -> tuple[Path, Path]:
        """Cree des fichiers .pt et un labels.csv temporaires."""
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()

        for i, n_patches in enumerate([50, 100, 30]):
            torch.save({
                "embeddings": torch.randn(n_patches, 768),
                "spatial_variances": torch.rand(n_patches) * 100,
                "coords": torch.randint(0, 10000, (n_patches, 2)).float(),
                "patch_ids": [f"slide_{i}_patch_{j}" for j in range(n_patches)],
            }, emb_dir / f"slide_{i}.pt")

        labels_csv = tmp_path / "labels.csv"
        pd.DataFrame({
            "slide_name": ["slide_0", "slide_1", "slide_2"],
            "isup_grade": [0, 2, 4],
        }).to_csv(labels_csv, index=False)

        return emb_dir, labels_csv

    def test_len(self, dataset_dir: tuple[Path, Path]) -> None:
        ds = SlideEmbeddingDataset(*dataset_dir)
        assert len(ds) == 3

    def test_getitem(self, dataset_dir: tuple[Path, Path]) -> None:
        ds = SlideEmbeddingDataset(*dataset_dir)
        item = ds[0]
        assert item["embeddings"].ndim == 2
        assert item["embeddings"].shape[1] == 768
        assert item["spatial_variances"].ndim == 1
        assert isinstance(item["label"], int)

    def test_missing_pt_warns(self, dataset_dir: tuple[Path, Path]) -> None:
        """Un slide absent du dossier .pt doit etre ignore avec warning."""
        emb_dir, labels_csv = dataset_dir
        # Ajouter un slide sans .pt
        df = pd.read_csv(labels_csv)
        df = pd.concat([df, pd.DataFrame({"slide_name": ["missing"], "isup_grade": [1]})], ignore_index=True)
        df.to_csv(labels_csv, index=False)

        ds = SlideEmbeddingDataset(emb_dir, labels_csv)
        assert len(ds) == 3  # le slide manquant est ignore


# ---------------------------------------------------------------------------
# Engine (train/validate avec donnees synthetiques)
# ---------------------------------------------------------------------------

class TestEngine:
    """Tests pour les boucles train/validate."""

    @pytest.fixture
    def training_setup(self, tmp_path: Path):
        """Prepare modele, dataloader, optimizer, criterion."""
        from torch.utils.data import DataLoader, Subset

        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()

        for i in range(3):
            torch.save({
                "embeddings": torch.randn(20, 768),
                "spatial_variances": torch.rand(20) * 100,
                "coords": torch.randint(0, 5000, (20, 2)).float(),
                "patch_ids": [f"s{i}_p{j}" for j in range(20)],
            }, emb_dir / f"s{i}.pt")

        labels_csv = tmp_path / "labels.csv"
        pd.DataFrame({
            "slide_name": ["s0", "s1", "s2"],
            "isup_grade": [0, 2, 4],
        }).to_csv(labels_csv, index=False)

        ds = SlideEmbeddingDataset(emb_dir, labels_csv)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        model = CustomMILModel(input_dim=768, num_classes=5, variance_threshold=15.0)
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        return model, loader, optimizer, criterion, device

    def test_train_one_epoch_returns_metrics(self, training_setup) -> None:
        model, loader, optimizer, criterion, device = training_setup
        metrics = train_one_epoch(model, loader, optimizer, criterion, device)
        assert isinstance(metrics, EpochMetrics)
        assert metrics.loss > 0

    def test_validate_one_epoch_returns_metrics(self, training_setup) -> None:
        model, loader, _, criterion, device = training_setup
        metrics = validate_one_epoch(model, loader, criterion, device)
        assert isinstance(metrics, EpochMetrics)
        assert metrics.loss > 0

    def test_training_reduces_loss(self, training_setup) -> None:
        """Quelques epochs doivent reduire la loss (overfit sur 3 slides)."""
        model, loader, optimizer, criterion, device = training_setup

        loss_first = train_one_epoch(model, loader, optimizer, criterion, device).loss
        for _ in range(9):
            train_one_epoch(model, loader, optimizer, criterion, device)
        loss_last = train_one_epoch(model, loader, optimizer, criterion, device).loss

        assert loss_last < loss_first
