"""Extract HistoEncoder embeddings from pre-extracted JPEG patches (e.g. SICAPv2).

Unlike the main extract_features.py (which reads patches from WSI files via
patch_registry CSVs), this script reads standalone JPEG patches from a directory
and groups them by slide ID (parsed from the filename prefix).

Pipeline per patch:
  JPEG (512x512) → Resize(224) → CenterCrop(224) → ToTensor
      → GPU Macenko Normalization → ImageNet Normalization
      → HistoEncoder backbone → [384] embedding

Output: one .pt file per slide containing:
  - embeddings:         [N, 384]
  - spatial_variances:  [N]       (Laplacian variance of each patch)
  - coords:             [N, 2]    (parsed from xini/yini in filename)
  - patch_ids:          list[str] (original filenames)
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────

def parse_slide_id(filename: str) -> str:
    """Extract the slide ID from a SICAPv2 filename.

    Example: '16B0001851_Block_Region_1_0_0_xini_6803_yini_59786.jpg'
             → '16B0001851'
    """
    return filename.split("_Block_")[0]


def parse_coords(filename: str) -> tuple[int, int]:
    """Extract (x, y) coordinates from xini/yini in filename."""
    x_match = re.search(r"xini_(\d+)", filename)
    y_match = re.search(r"yini_(\d+)", filename)
    x = int(x_match.group(1)) if x_match else 0
    y = int(y_match.group(1)) if y_match else 0
    return x, y


def compute_laplacian_variance(image: Image.Image) -> float:
    """Compute Laplacian spatial variance (focus/texture metric)."""
    gray = np.array(image.convert("L"), dtype=np.float64)
    # Simple Laplacian kernel convolution
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] +
        gray[1:-1, :-2] + gray[1:-1, 2:] -
        4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


# ── Dataset ──────────────────────────────────────────────────────────

class JPEGPatchDataset(Dataset):
    """Reads JPEG patches from a directory and crops them to match training scale.
    
    PANDA was 20x magnification (256x256), SICAPv2 is 10x (512x512).
    To match the physical tissue scale the network was trained on, we must
    extract 128x128 sub-patches from the 10x image, which are then upscaled 
    to 224x224 by the transform.
    """

    def __init__(
        self,
        image_paths: list[Path],
        transform: T.Compose | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform
        
        # 512x512 split into 128x128 crops yields a 4x4 grid (16 crops)
        self.crop_size = 128
        self.crops_per_image = 16
        
        # Simple per-worker manual cache to avoid reading the same JPEG 16 times
        self._cache_path = None
        self._cache_img = None

    def __len__(self) -> int:
        return len(self.image_paths) * self.crops_per_image

    def __getitem__(self, idx: int) -> dict:
        img_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image
        
        path = self.image_paths[img_idx]
        
        if self._cache_path != path:
            self._cache_img = Image.open(path).convert("RGB")
            self._cache_path = path
            
        img = self._cache_img
        
        width, height = img.size
        grid_width = width // self.crop_size
        
        # Guard against weirdly sized images gracefully
        grid_width = max(1, grid_width)
        
        grid_x = crop_idx % grid_width
        grid_y = crop_idx // grid_width
        
        x = grid_x * self.crop_size
        y = grid_y * self.crop_size
        
        # Bounds logic just in case image is slightly smaller than expected
        x = max(0, min(x, width - self.crop_size))
        y = max(0, min(y, height - self.crop_size))
        
        crop = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        variance = compute_laplacian_variance(crop)
        
        base_x, base_y = parse_coords(path.name)

        if self.transform:
            crop_tensor = self.transform(crop)
        else:
            crop_tensor = T.ToTensor()(crop)

        return {
            "image": crop_tensor,
            "patch_id": f"{path.stem}_crop{crop_idx}",
            "slide_id": parse_slide_id(path.name),
            "coord_x": float(base_x + x),
            "coord_y": float(base_y + y),
            "variance_spatiale": variance,
        }


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HistoEncoder embeddings from pre-extracted JPEG patches."
    )
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Directory containing JPEG patches.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save per-slide .pt files.")
    parser.add_argument("--backbone-name", type=str, default="histoencoder",
                        choices=["histoencoder", "virchow2"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    from prostate_mil.models.backbones.backbone import (
        FrozenBackboneExtractor,
        get_image_size,
        load_foundation_backbone,
    )
    from prostate_mil.data.preprocessing.color_normalization import TorchMacenkoNormalizerWrapper

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.device in ("auto", "cuda"):
        dev = torch.device("cuda")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("⚠️  No GPU — running on CPU")

    use_amp = dev.type == "cuda"

    # ── Discover patches ─────────────────────────────────────────────
    all_jpgs = sorted(args.images_dir.glob("*.jpg"))
    if not all_jpgs:
        print(f"❌ No .jpg files found in {args.images_dir}")
        return

    # Group by slide ID and filter already done
    slide_patches: dict[str, list[Path]] = defaultdict(list)
    for p in all_jpgs:
        slide_id = parse_slide_id(p.name)
        slide_patches[slide_id].append(p)

    # Filter already-extracted slides
    todo: dict[str, list[Path]] = {}
    skipped = 0
    for slide_id, patches in slide_patches.items():
        if (args.output_dir / f"{slide_id}.pt").exists():
            skipped += 1
        else:
            todo[slide_id] = patches

    total_slides = len(slide_patches)
    total_patches = sum(len(v) for v in todo.values())

    print(f"📦 Found {total_slides} slides, {len(all_jpgs)} total patches")
    print(f"   Already done: {skipped} | To process: {len(todo)} ({total_patches} patches)")

    if not todo:
        print("✅ All slides already extracted!")
        return

    # ── Load backbone ────────────────────────────────────────────────
    image_size = get_image_size(args.backbone_name)
    backbone = load_foundation_backbone(args.backbone_name)
    extractor = FrozenBackboneExtractor(backbone).to(dev)
    gpu_normalizer = TorchMacenkoNormalizerWrapper(device=dev)
    imagenet_norm = T.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )

    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])

    print(f"🧠 Backbone: {args.backbone_name} | Image size: {image_size}")
    print(f"{'='*60}")

    # ── Process globally to keep GPU fed ──────────────────────────────────────────
    start_time = time.time()
    slides_done = 0
    patches_done = 0

    all_patch_paths = []
    for paths in todo.values():
        all_patch_paths.extend(paths)

    dataset = JPEGPatchDataset(all_patch_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        prefetch_factor=2,
    )

    current_slide = None
    slide_data = {"embeddings": [], "spatial_variances": [], "coords": [], "patch_ids": []}

    def save_current_slide():
        if not slide_data["embeddings"]:
            return 0
        result = {
            "embeddings": torch.cat(slide_data["embeddings"], dim=0),
            "spatial_variances": torch.tensor(slide_data["spatial_variances"], dtype=torch.float32),
            "coords": torch.tensor(slide_data["coords"], dtype=torch.float32),
            "patch_ids": slide_data["patch_ids"],
        }
        torch.save(result, args.output_dir / f"{current_slide}.pt")
        return result["embeddings"].shape[0]

    for batch in tqdm(loader, desc="Extraction", unit="batch"):
        images = batch["image"].to(dev, non_blocking=True)

        # GPU pipeline: Macenko → ImageNet norm → Backbone
        images = gpu_normalizer.transform(images)
        images = imagenet_norm(images)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            embeddings = extractor(images)

        embeddings = embeddings.float().cpu()
        variances = batch["variance_spatiale"].tolist()
        coords_x = batch["coord_x"].tolist()
        coords_y = batch["coord_y"].tolist()
        patch_ids = batch["patch_id"]
        slide_ids = batch["slide_id"]

        for i in range(len(embeddings)):
            sid = slide_ids[i]

            if current_slide is None:
                current_slide = sid

            if sid != current_slide:
                patches_done += save_current_slide()
                slides_done += 1
                
                # Reset for new slide
                current_slide = sid
                slide_data = {"embeddings": [], "spatial_variances": [], "coords": [], "patch_ids": []}

            slide_data["embeddings"].append(embeddings[i:i+1])
            slide_data["spatial_variances"].append(variances[i])
            slide_data["coords"].append((coords_x[i], coords_y[i]))
            slide_data["patch_ids"].append(patch_ids[i])

    # Save the last slide
    if current_slide is not None:
        patches_done += save_current_slide()
        slides_done += 1

    elapsed = time.time() - start_time
    print(f"\n✅ Done! {slides_done} slides, {patches_done:,} patches in {elapsed:.0f}s")
    print(f"   Output: {args.output_dir}")


if __name__ == "__main__":
    main()
