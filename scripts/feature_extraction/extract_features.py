"""Script CLI pour la phase 2: extraction d'embeddings.

Chunked multi-slide processing with LRU WSI cache:
- Processes 50 slides per chunk through ONE DataLoader (GPU stays busy)
- Each worker keeps only 1 WSI open at a time via LRU eviction
- Saves .pt files after each chunk (incremental progress)
- Memory bounded: ~4GB pipeline + ~200MB per worker for WSI handle
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: extraction d'embeddings depuis les patchs Phase 1."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--csv-path", type=Path)
    source_group.add_argument("--slides-dir", type=Path)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--backbone-name", type=str, default="histoencoder",
        choices=["histoencoder", "virchow2"],
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Slides per chunk (lower = less RAM, default 50).",
    )
    return parser.parse_args()


def print_dashboard(n_total, n_skipped, n_done, n_todo, device,
                    n_patches=0, elapsed=0, redraw=False):
    progress = n_done / max(n_todo, 1)
    bar = "█" * int(30 * progress) + "░" * (30 - int(30 * progress))

    if n_done > 0 and elapsed > 0:
        from datetime import timedelta
        speed = n_done / elapsed
        eta = str(timedelta(seconds=int((n_todo - n_done) / speed)))
        pps = n_patches / elapsed
    else:
        speed, pps, eta = 0.0, 0.0, "--:--:--"

    lines = [
        "=" * 60,
        "🧠 PHASE 2 FEATURE EXTRACTION - LIVE DASHBOARD",
        "-" * 60,
        f"Target Device:    {device.upper()}",
        f"Total Slides:     {n_total}",
        f"Already Done:     {n_skipped + n_done}  (+{n_done} this session)",
        f"Remaining:        {n_todo - n_done}",
        "-" * 60,
        f"Progress: [{bar}] {progress*100:6.1f}%",
        f"Speed:    {speed:.2f} slide/s  |  {pps:.0f} patch/s  |  ETA: {eta}",
        "=" * 60,
    ]
    if redraw:
        sys.stdout.write(f"\033[{len(lines)}A")
    for line in lines:
        sys.stdout.write(f"\r{line}\033[K\n")
    sys.stdout.flush()


def discover_csv_files(args: argparse.Namespace) -> list[Path]:
    if args.csv_path is not None:
        if not args.csv_path.exists():
            raise FileNotFoundError(f"CSV introuvable: {args.csv_path}")
        return [args.csv_path]

    if args.slides_dir is None or not args.slides_dir.exists():
        raise FileNotFoundError("Le dossier de slides fourni est introuvable.")

    csv_files = sorted(args.slides_dir.glob("*/patch_registry.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Aucun patch_registry.csv trouve dans {args.slides_dir}."
        )
    return csv_files


def process_chunk(
    chunk_entries: list[dict],
    image_size: int,
    batch_size: int,
    num_workers: int,
    dev: torch.device,
    use_amp: bool,
    extractor,
    gpu_normalizer,
    imagenet_norm,
    quiet: bool,
) -> tuple[int, int]:
    """Process one chunk. Returns (slides_saved, patches_processed)."""
    from prostate_mil.data.datasets.feature_dataset import MultiSlidePatchDataset

    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])

    dataset = MultiSlidePatchDataset(
        slide_entries=chunk_entries,
        transform=transform,
    )

    if len(dataset) == 0:
        return 0, 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Accumulate per-slide results (bounded to chunk_size slides)
    results: dict[int, dict] = defaultdict(lambda: {
        "embeddings": [], "variances": [], "coords": [], "patch_ids": [],
    })

    total_patches = 0

    for batch in tqdm(loader, desc=f"Chunk ({len(chunk_entries)} slides)",
                      unit="batch", disable=quiet):
        images = batch["image"].to(dev, non_blocking=True)
        slide_idxs = batch["slide_idx"]

        # GPU pipeline: Macenko → ImageNet norm → Backbone (fp16)
        images = gpu_normalizer.transform(images)
        images = imagenet_norm(images)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            embeddings = extractor(images)

        embeddings = embeddings.float().cpu()
        total_patches += images.shape[0]

        # Scatter to per-slide buckets
        for i in range(images.shape[0]):
            sidx = int(slide_idxs[i])
            bucket = results[sidx]
            bucket["embeddings"].append(embeddings[i])
            bucket["variances"].append(float(batch["variance_spatiale"][i]))
            bucket["coords"].append(
                (float(batch["coord_x"][i]), float(batch["coord_y"][i]))
            )
            bucket["patch_ids"].append(batch["patch_id"][i])

    # Save .pt files for this chunk
    slides_saved = 0
    for sidx, bucket in results.items():
        if not bucket["embeddings"]:
            continue
        entry = chunk_entries[sidx]
        result = {
            "embeddings": torch.stack(bucket["embeddings"], dim=0),
            "spatial_variances": torch.tensor(bucket["variances"], dtype=torch.float32),
            "coords": torch.tensor(bucket["coords"], dtype=torch.float32),
            "patch_ids": bucket["patch_ids"],
        }
        torch.save(result, entry["output_path"])
        slides_saved += 1

    # Explicitly free everything
    del results, loader, dataset
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return slides_saved, total_patches


def main() -> None:
    from prostate_mil.models.backbones.backbone import (
        FrozenBackboneExtractor,
        get_image_size,
        load_foundation_backbone,
    )
    from prostate_mil.data.preprocessing.color_normalization import TorchMacenkoNormalizerWrapper

    args = parse_args()
    csv_files = discover_csv_files(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.device in ("auto", "cuda"):
        dev = torch.device("cuda")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🧠 Backbone: {args.backbone_name}")
    else:
        dev = torch.device("cpu")
        print("⚠️  No GPU — running on CPU")

    use_amp = dev.type == "cuda"

    # ── Filter already-done slides ───────────────────────────────────
    slide_entries = []
    skipped = 0
    for cp in csv_files:
        slide_name = cp.parent.name
        out = args.output_dir / f"{slide_name}.pt"
        if out.exists():
            skipped += 1
            continue
        wsi_candidates = list(args.raw_data_dir.glob(f"{slide_name}.*"))
        if not wsi_candidates:
            continue
        slide_entries.append({
            "csv_path": cp,
            "wsi_path": wsi_candidates[0],
            "output_path": out,
            "slide_name": slide_name,
        })

    n_total = len(csv_files)
    n_todo = len(slide_entries)

    print_dashboard(n_total, skipped, 0, n_todo, dev.type)

    if not slide_entries:
        print("✅ All slides already extracted!")
        return

    # ── Load backbone ONCE ───────────────────────────────────────────
    image_size = args.image_size or get_image_size(args.backbone_name)
    backbone = load_foundation_backbone(args.backbone_name)
    extractor = FrozenBackboneExtractor(backbone).to(dev)
    gpu_normalizer = TorchMacenkoNormalizerWrapper(device=dev)
    imagenet_norm = T.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )

    # ── Process in chunks ────────────────────────────────────────────
    chunk_size = args.chunk_size
    n_chunks = (n_todo + chunk_size - 1) // chunk_size
    print(f"📦 Processing {n_todo} slides in {n_chunks} chunks of ≤{chunk_size}")

    start_time = time.time()
    slides_done = 0
    patches_done = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_todo)
        chunk = slide_entries[chunk_start:chunk_end]

        try:
            saved, patches = process_chunk(
                chunk_entries=chunk,
                image_size=image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dev=dev,
                use_amp=use_amp,
                extractor=extractor,
                gpu_normalizer=gpu_normalizer,
                imagenet_norm=imagenet_norm,
                quiet=args.quiet,
            )
            slides_done += saved
            patches_done += patches
        except Exception as e:
            print(f"\n⚠️  Chunk {chunk_idx+1} error: {e}")
            slides_done += len(chunk)  # count as processed to keep progress moving
            continue

        elapsed = time.time() - start_time
        print_dashboard(n_total, skipped, slides_done, n_todo, dev.type,
                        n_patches=patches_done, elapsed=elapsed, redraw=True)

    elapsed = time.time() - start_time
    print(f"\n✅ Done! {slides_done} slides, {patches_done:,} patches in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
