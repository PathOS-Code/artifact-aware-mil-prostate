"""Configurations par defaut pour les quatre phases du pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration de la phase 1."""

    patch_size: int = 256
    mask_level: int | None = None
    max_mask_dimension: int = 4096
    tissue_threshold: float = 0.15
    min_tissue_component_area: int = 128
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/registries")


@dataclass(slots=True)
class FeatureExtractionConfig:
    """Configuration de la phase 2.

    Backbones disponibles (embedding_dim / image_size) :
        - histoencoder : 384 / 224  (XCiT-S12, prostate-optimized)
        - virchow2     : 1280 / 224 (ViT-H/14, histopath, necessite HF token)
    """

    image_size: int = 224
    embedding_dim: int = 1280
    batch_size: int = 128
    num_workers: int = 4
    backbone_name: str = "virchow2"
    device: str = "auto"
    embeddings_dir: Path = Path("outputs/embeddings")


@dataclass(slots=True)
class GatingConfig:
    """Configuration du module de gating de la phase 3."""

    input_dim: int = 1280
    attention_dim: int = 256
    aggregator: str = "abmil"  # 'abmil' ou 'transmil'
    # Calibrated on 200 slides: P10 of Laplacian spatial variance = 122.
    # Catches bottom ~10% lowest-variance patches as artifact candidates.
    # Only flagged if ALSO high L2 norm (dual condition).
    variance_threshold: float = 122.0
    alpha: float = 1e-4
    z_score_factor: float = 2.0


@dataclass(slots=True)
class TrainingConfig:
    """Configuration d'entrainement de la phase 4."""

    num_classes: int = 6  # ISUP grades 0-5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    loss_type: str = "focal"  # 'focal' (categorical) or 'coral' (ordinal)
    epochs: int = 50
    checkpoint_dir: Path = Path("outputs/checkpoints")


@dataclass(slots=True)
class HeatmapConfig:
    """Configuration de generation de heatmaps cliniques."""

    thumbnail_level: int = 2
    colormap: int = 2
    output_dir: Path = Path("outputs/reports")


@dataclass
class ProjectConfig:
    """Configuration composite de l'ensemble du pipeline.

    Automatically syncs ``gating.input_dim`` from
    ``feature_extraction.embedding_dim`` so the dimension is set once.
    """

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_extraction: FeatureExtractionConfig = field(
        default_factory=FeatureExtractionConfig
    )
    gating: GatingConfig = field(default_factory=GatingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)

    def __post_init__(self) -> None:
        self.gating.input_dim = self.feature_extraction.embedding_dim
