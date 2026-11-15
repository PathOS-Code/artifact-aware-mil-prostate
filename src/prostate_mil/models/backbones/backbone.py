"""Chargement d'un backbone visuel gele pour extraction d'embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

_BACKBONE_REGISTRY: dict[str, callable] = {}

# Dimension d'embedding par backbone — utilisee par le reste du pipeline.
BACKBONE_EMBEDDING_DIM: dict[str, int] = {}

# Taille d'image attendue par chaque backbone.
BACKBONE_IMAGE_SIZE: dict[str, int] = {}


def _register(name: str, embedding_dim: int, image_size: int = 224):
    """Decorateur pour enregistrer un backbone avec ses metadonnees."""

    def wrapper(fn):
        _BACKBONE_REGISTRY[name] = fn
        BACKBONE_EMBEDDING_DIM[name] = embedding_dim
        BACKBONE_IMAGE_SIZE[name] = image_size
        return fn

    return wrapper


# ── HistoEncoder (prostate, XCiT-S12) ───────────────────────────────


_HISTOENCODER_WEIGHTS_URL = (
    "https://dl.dropboxusercontent.com/s/tbff9wslc8p7ie3/prostate_small.pth?dl=0"
)


@_register("histoencoder", embedding_dim=384, image_size=224)
def _load_histoencoder() -> nn.Module:
    """HistoEncoder prostate_small — XCiT-S12 pre-entraine sur tissus prostatiques.

    Charge l'architecture XCiT-S12 via timm et applique les poids
    HistoEncoder (entraines sur du tissu prostatique).
    Sortie: [B, 384].
    """
    import timm

    model = timm.create_model("xcit_small_12_p16_224", num_classes=0)
    state_dict = torch.hub.load_state_dict_from_url(
        _HISTOENCODER_WEIGHTS_URL, map_location="cpu",
    )
    # Les poids HistoEncoder contiennent une tete de classification
    # qu'on supprime puisqu'on utilise num_classes=0.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    model.load_state_dict(state_dict, strict=False)
    return model


# ── Virchow2 (Paige / Microsoft, ViT-H/14) ─────────────────────────


class _Virchow2Wrapper(nn.Module):
    """Encapsule Virchow2 (timm hub) et extrait le CLS token [B, 1280].

    La sortie brute du modele est ``[B, 261, 1280]``:
      - token 0     : CLS token
      - tokens 1-4  : register tokens (ignores)
      - tokens 5-260: patch tokens

    On retourne le CLS token seul pour rester coherent avec un pipeline
    ou chaque patch produit un unique vecteur d'embedding.
    """

    def __init__(self) -> None:
        super().__init__()
        import timm
        from timm.layers import SwiGLUPacked

        local_model = _load_virchow2_from_hf_cache(timm, SwiGLUPacked)
        if local_model is not None:
            self.model = local_model
        else:
            self.model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        self.model.eval()

    def forward(self, batch: Tensor) -> Tensor:
        output = self.model(batch)  # [B, 261, 1280]
        return output[:, 0]  # CLS token -> [B, 1280]


@_register("virchow2", embedding_dim=1280, image_size=224)
def _load_virchow2() -> nn.Module:
    """Virchow2 — ViT-H/14 pre-entraine sur 3.1M lames histopathologiques."""
    return _Virchow2Wrapper()


def _load_virchow2_from_hf_cache(timm_module, swiglu_layer) -> nn.Module | None:
    """Load Virchow2 directly from the local HF cache when available.

    Some Windows environments crash in the ``hf-hub:`` resolver with
    ``OPENSSL_Uplink ... no OPENSSL_Applink``. The cached snapshot contains a
    standard timm config and safetensors checkpoint, so this constructs the same
    architecture and loads the same weights without invoking that resolver.
    """

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--paige-ai--Virchow2"
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        snapshots_dir.glob("*/model.safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    try:
        from safetensors.torch import load_file
    except ImportError:
        return None

    weights_path = candidates[0]
    logger.info("Loading Virchow2 from local HF cache: %s", weights_path)
    model = timm_module.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        img_size=224,
        init_values=1e-5,
        num_classes=0,
        reg_tokens=4,
        mlp_ratio=5.3375,
        global_pool="",
        dynamic_img_size=True,
        mlp_layer=swiglu_layer,
        act_layer=torch.nn.SiLU,
    )
    state_dict = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


# ── API publique ─────────────────────────────────────────────────────


def load_foundation_backbone(backbone_name: str = "histoencoder") -> nn.Module:
    """Charge un backbone pre-entraine et le bascule en mode inference.

    Args:
        backbone_name: Cle du backbone dans le registre.
            Backbones disponibles : histoencoder, virchow2.

    Returns:
        nn.Module produisant des embeddings [B, D].
    """

    if backbone_name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Backbone inconnu: '{backbone_name}'. "
            f"Disponibles: {list(_BACKBONE_REGISTRY.keys())}"
        )

    logger.info("Chargement du backbone: %s", backbone_name)
    model = _BACKBONE_REGISTRY[backbone_name]()
    model.eval()
    return model


def get_embedding_dim(backbone_name: str) -> int:
    """Retourne la dimension d'embedding pour un backbone donne."""

    if backbone_name not in BACKBONE_EMBEDDING_DIM:
        raise ValueError(
            f"Backbone inconnu: '{backbone_name}'. "
            f"Disponibles: {list(BACKBONE_EMBEDDING_DIM.keys())}"
        )
    return BACKBONE_EMBEDDING_DIM[backbone_name]


def get_image_size(backbone_name: str) -> int:
    """Retourne la taille d'image attendue pour un backbone donne."""

    if backbone_name not in BACKBONE_IMAGE_SIZE:
        raise ValueError(
            f"Backbone inconnu: '{backbone_name}'. "
            f"Disponibles: {list(BACKBONE_IMAGE_SIZE.keys())}"
        )
    return BACKBONE_IMAGE_SIZE[backbone_name]


class FrozenBackboneExtractor(nn.Module):
    """Encapsule un backbone gele utilise avec `torch.no_grad()`."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Projette un batch de patchs en embeddings [B, D]."""

        with torch.no_grad():
            return self.backbone(batch)
