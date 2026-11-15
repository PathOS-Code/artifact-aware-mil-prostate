"""Transformations standardisees pour backbone de fondation."""

from __future__ import annotations

from torchvision import transforms

# Parametres de normalisation par backbone.
# ViT ImageNet et HistoEncoder utilisent les stats ImageNet classiques.
# Virchow2 utilise les memes stats ImageNet (specifie dans son README HuggingFace).
_NORMALIZE_PARAMS: dict[str, dict[str, tuple[float, ...]]] = {
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}

_BACKBONE_NORMALIZE_KEY: dict[str, str] = {
    "vit_base_patch16_224": "imagenet",
    "histoencoder": "imagenet",
    "virchow2": "imagenet",
}


def build_feature_transforms(
    image_size: int = 224,
    backbone_name: str = "vit_base_patch16_224",
) -> transforms.Compose:
    """Construit la chaine Resize -> CenterCrop -> Normalize.

    La normalisation est adaptee au backbone selectionne.
    """

    norm_key = _BACKBONE_NORMALIZE_KEY.get(backbone_name, "imagenet")
    norm_params = _NORMALIZE_PARAMS[norm_key]

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"]),
        ]
    )
