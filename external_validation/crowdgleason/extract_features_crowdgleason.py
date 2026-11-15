"""Extract Virchow2/HistoEncoder bags for CrowdGleason external evaluation.

This script mirrors the SICAPv2 external-patch flow: standalone JPEG patches
are grouped into slide-level bags, transformed with the same backbone pipeline,
and saved as one .pt file per slide with keys consumed by SlideEmbeddingDataset:

  embeddings, spatial_variances, coords, patch_ids

CrowdGleason is described by Zenodo as 512x512 patches at 10x magnification.
If the installed images have a different pixel size, the default behavior is to
resize each image to a canonical 512x512 before applying the 10x -> 20x physical
scale correction used by the SICAPv2 script. Set --canonical-patch-size 0 to
disable that resize.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def compute_laplacian_variance(image: Image.Image) -> float:
    gray = np.array(image.convert("L"), dtype=np.float64)
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


class CrowdGleasonCropDataset(Dataset):
    def __init__(
        self,
        patch_rows: list[dict],
        crop_size: int,
        transform: T.Compose,
        canonical_patch_size: int | None,
    ) -> None:
        self.patch_rows = patch_rows
        self.crop_size = crop_size
        self.transform = transform
        self.canonical_patch_size = canonical_patch_size

        self._cache_path: Path | None = None
        self._cache_img: Image.Image | None = None
        self._cache_grid: tuple[int, int] | None = None

        self._index: list[tuple[int, int, int]] = []
        for patch_idx, row in enumerate(self.patch_rows):
            width = int(row["width"])
            height = int(row["height"])
            if canonical_patch_size is not None:
                width = canonical_patch_size
                height = canonical_patch_size
            grid_x = max(1, width // crop_size)
            grid_y = max(1, height // crop_size)
            for crop_y in range(grid_y):
                for crop_x in range(grid_x):
                    self._index.append((patch_idx, crop_x, crop_y))

    def __len__(self) -> int:
        return len(self._index)

    def _load_image(self, path: Path) -> tuple[Image.Image, tuple[int, int]]:
        if self._cache_path != path:
            img = Image.open(path).convert("RGB")
            if self.canonical_patch_size is not None and img.size != (
                self.canonical_patch_size,
                self.canonical_patch_size,
            ):
                img = img.resize(
                    (self.canonical_patch_size, self.canonical_patch_size),
                    resample=Image.Resampling.BILINEAR,
                )
            self._cache_path = path
            self._cache_img = img
            self._cache_grid = (
                max(1, img.size[0] // self.crop_size),
                max(1, img.size[1] // self.crop_size),
            )
        assert self._cache_img is not None
        assert self._cache_grid is not None
        return self._cache_img, self._cache_grid

    def __getitem__(self, index: int) -> dict:
        patch_idx, crop_x_idx, crop_y_idx = self._index[index]
        row = self.patch_rows[patch_idx]
        path = Path(row["patch_path"])
        img, _ = self._load_image(path)

        x = min(crop_x_idx * self.crop_size, max(0, img.size[0] - self.crop_size))
        y = min(crop_y_idx * self.crop_size, max(0, img.size[1] - self.crop_size))
        crop = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        variance = compute_laplacian_variance(crop)

        return {
            "image": self.transform(crop),
            "slide_id": str(row["slide_id"]),
            "patch_id": f"{row['patch_id']}_cropx{crop_x_idx}_cropy{crop_y_idx}",
            "coord_x": float(row["coord_x"] + x),
            "coord_y": float(row["coord_y"] + y),
            "variance_spatiale": variance,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CrowdGleason slide bags from standalone JPEG patches."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("external_validation/labels/crowdgleason_patch_manifest.csv"),
        help="Patch manifest produced by prepare_crowdgleason_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/external_embeddings/crowdgleason/virchow2"),
        help="Directory to save per-slide .pt bag files.",
    )
    parser.add_argument(
        "--backbone-name",
        default="virchow2",
        choices=["virchow2", "histoencoder"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-magnification", type=float, default=10.0)
    parser.add_argument("--target-magnification", type=float, default=20.0)
    parser.add_argument(
        "--training-patch-size",
        type=int,
        default=256,
        help="PANDA patch size at target magnification.",
    )
    parser.add_argument(
        "--canonical-patch-size",
        type=int,
        default=512,
        help="Resize CrowdGleason patch images to this size first. Use 0 to disable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute slides even if the .pt output already exists.",
    )
    return parser.parse_args()


def main() -> None:
    from prostate_mil.data.preprocessing.color_normalization import (
        TorchMacenkoNormalizerWrapper,
    )
    from prostate_mil.models.backbones.backbone import (
        FrozenBackboneExtractor,
        get_image_size,
        load_foundation_backbone,
    )

    args = parse_args()
    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"Patch manifest not found: {args.manifest_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv)
    required = {"slide_id", "patch_id", "patch_path", "coord_x", "coord_y"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    missing_paths = [Path(p) for p in manifest["patch_path"] if not Path(p).exists()]
    if missing_paths:
        print(f"WARNING: {len(missing_paths)} missing patch files; they will be skipped.")
        for path in missing_paths[:10]:
            print(f"  missing: {path}")
        manifest = manifest[manifest["patch_path"].map(lambda p: Path(p).exists())]

    size_counts: Counter[str] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    for patch_path in tqdm(manifest["patch_path"].tolist(), desc="Inspecting image sizes"):
        with Image.open(patch_path) as img:
            widths.append(img.size[0])
            heights.append(img.size[1])
            size_counts[f"{img.size[0]}x{img.size[1]}"] += 1
    manifest = manifest.copy()
    manifest["width"] = widths
    manifest["height"] = heights

    canonical = args.canonical_patch_size if args.canonical_patch_size > 0 else None
    crop_size = int(round(args.training_patch_size * args.source_magnification / args.target_magnification))
    crop_size = max(1, crop_size)

    slide_patches: dict[str, list[dict]] = defaultdict(list)
    for row in manifest.to_dict("records"):
        out_path = args.output_dir / f"{row['slide_id']}.pt"
        if args.force or not out_path.exists():
            slide_patches[str(row["slide_id"])].append(row)

    total_existing = manifest["slide_id"].nunique() - len(slide_patches)
    total_crops = 0
    for rows in slide_patches.values():
        for row in rows:
            width = canonical if canonical is not None else int(row["width"])
            height = canonical if canonical is not None else int(row["height"])
            total_crops += max(1, width // crop_size) * max(1, height // crop_size)

    print("CrowdGleason feature extraction")
    print(f"  Backbone: {args.backbone_name}")
    print(f"  Image sizes observed: {dict(size_counts)}")
    print(f"  Canonical resize: {canonical if canonical is not None else 'disabled'}")
    print(f"  Crop size for {args.source_magnification:g}x -> {args.target_magnification:g}x: {crop_size}")
    print(f"  Slides already done: {total_existing}")
    print(f"  Slides to process: {len(slide_patches)}")
    print(f"  Crops to embed: {total_crops:,}")

    if not slide_patches:
        print("All CrowdGleason slide bags already exist.")
        return

    if torch.cuda.is_available() and args.device in ("auto", "cuda"):
        dev = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu" if args.device == "auto" else args.device)
        print(f"  Device: {dev}")

    use_amp = dev.type == "cuda"
    image_size = get_image_size(args.backbone_name)
    backbone = load_foundation_backbone(args.backbone_name)
    extractor = FrozenBackboneExtractor(backbone).to(dev)
    gpu_normalizer = TorchMacenkoNormalizerWrapper(device=dev)
    imagenet_norm = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    transform = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])

    all_rows: list[dict] = []
    for slide_id in sorted(slide_patches):
        all_rows.extend(slide_patches[slide_id])

    dataset = CrowdGleasonCropDataset(
        patch_rows=all_rows,
        crop_size=crop_size,
        transform=transform,
        canonical_patch_size=canonical,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": dev.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    current_slide: str | None = None
    slide_data = {"embeddings": [], "spatial_variances": [], "coords": [], "patch_ids": []}

    def save_current_slide() -> int:
        if current_slide is None or not slide_data["embeddings"]:
            return 0
        result = {
            "embeddings": torch.cat(slide_data["embeddings"], dim=0),
            "spatial_variances": torch.tensor(slide_data["spatial_variances"], dtype=torch.float32),
            "coords": torch.tensor(slide_data["coords"], dtype=torch.float32),
            "patch_ids": list(slide_data["patch_ids"]),
        }
        torch.save(result, args.output_dir / f"{current_slide}.pt")
        return int(result["embeddings"].shape[0])

    start = time.time()
    slides_done = 0
    crops_done = 0
    for batch in tqdm(loader, desc="Embedding CrowdGleason", unit="batch"):
        images = batch["image"].to(dev, non_blocking=True)
        images = gpu_normalizer.transform(images)
        images = imagenet_norm(images)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            embeddings = extractor(images)
        embeddings = embeddings.float().cpu()

        slide_ids = list(batch["slide_id"])
        patch_ids = list(batch["patch_id"])
        variances = batch["variance_spatiale"].tolist()
        coord_x = batch["coord_x"].tolist()
        coord_y = batch["coord_y"].tolist()

        for i, slide_id in enumerate(slide_ids):
            if current_slide is None:
                current_slide = str(slide_id)
            if str(slide_id) != current_slide:
                crops_done += save_current_slide()
                slides_done += 1
                current_slide = str(slide_id)
                slide_data = {"embeddings": [], "spatial_variances": [], "coords": [], "patch_ids": []}
                gc.collect()

            slide_data["embeddings"].append(embeddings[i : i + 1])
            slide_data["spatial_variances"].append(float(variances[i]))
            slide_data["coords"].append((float(coord_x[i]), float(coord_y[i])))
            slide_data["patch_ids"].append(str(patch_ids[i]))

    crops_done += save_current_slide()
    if current_slide is not None:
        slides_done += 1

    elapsed = time.time() - start
    print(f"Done: {slides_done} slides, {crops_done:,} crops in {elapsed:.0f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
