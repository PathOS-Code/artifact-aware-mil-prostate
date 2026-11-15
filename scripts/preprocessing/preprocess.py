"""Script CLI pour la phase 1: pretraitement et extraction de patchs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SUPPORTED_WSI_SUFFIXES = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".bif",
    ".vms",
    ".vmu",
}


def parse_args() -> argparse.Namespace:
    """Construit et parse les arguments de la CLI."""

    parser = argparse.ArgumentParser(
        description="Phase 1: segmentation tissu, extraction de patchs, Macenko et CSV."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--slide-path", type=Path, help="Chemin vers une WSI unique.")
    source_group.add_argument(
        "--slides-dir",
        type=Path,
        help="Dossier contenant plusieurs lames WSI a preprocesser.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Dossier racine de sortie pour les slides preprocesses.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Taille des patchs extraits au niveau 0.",
    )
    parser.add_argument(
        "--mask-level",
        type=int,
        default=None,
        help="Niveau OpenSlide pour le masque tissu. Par defaut: selection automatique.",
    )
    parser.add_argument(
        "--max-mask-dimension",
        type=int,
        default=4096,
        help="Taille max cible pour choisir automatiquement un niveau de masque compact.",
    )
    parser.add_argument(
        "--tissue-threshold",
        type=float,
        default=0.15,
        help="Couverture minimale de tissu requise dans une fenetre candidate.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Explore recursivement le dossier de lames.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Nombre de processus en parallele (Phase 1 est CPU-bound).",
    )
    return parser.parse_args()


def discover_slides(args: argparse.Namespace) -> list[Path]:
    """Retourne la liste des WSI a traiter."""

    if args.slide_path is not None:
        return [args.slide_path]

    if args.slides_dir is None or not args.slides_dir.exists():
        raise FileNotFoundError("Le dossier de lames fourni est introuvable.")

    pattern = "**/*" if args.recursive else "*"
    slides = sorted(
        path
        for path in args.slides_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_WSI_SUFFIXES
    )

    if not slides:
        raise FileNotFoundError(
            f"Aucune WSI compatible trouvee dans {args.slides_dir}."
        )
    return slides


def build_config(args: argparse.Namespace) -> "PreprocessingConfig":
    """Construit un PreprocessingConfig depuis les arguments CLI."""

    from prostate_mil.config.defaults import PreprocessingConfig

    return PreprocessingConfig(
        patch_size=args.patch_size,
        mask_level=args.mask_level,
        max_mask_dimension=args.max_mask_dimension,
        tissue_threshold=args.tissue_threshold,
        processed_data_dir=args.output_dir,
    )


def preprocess_slide(slide_path: Path, config: "PreprocessingConfig") -> Path:
    """Traite une slide, sauve les patchs et produit le registre CSV."""

    from prostate_mil.data.patches.patch_extraction import PatchExtractor
    from prostate_mil.data.datasets.registry import save_patch_registry

    slide_output_dir = config.processed_data_dir / slide_path.stem
    extractor = PatchExtractor.from_config(config)

    # Note: On desactive tqdm interne si on est en parallele pour eviter le chaos visuel
    records, slide_info = extractor.extract_and_save_patches(
        slide_path=slide_path,
        output_dir=config.processed_data_dir,
        show_progress=False, 
    )
    csv_path = save_patch_registry(records, slide_output_dir / "patch_registry.csv")
    return len(records)


def print_stats(n_total, n_skipped, n_done, n_todo, num_workers, elapsed_time=0, redraw=False):
    # Calcul d'un petit bar de progression manuel [####....]
    bar_width = 30
    progress = n_done / max(n_todo, 1)
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    percent = progress * 100
    
    # Calcul du temps restant estimé (ETA)
    if n_done > 0:
        it_per_sec = n_done / elapsed_time
        eta_seconds = (n_todo - n_done) / it_per_sec
        from datetime import timedelta
        eta_str = str(timedelta(seconds=int(eta_seconds)))
    else:
        eta_str = "--:--:--"

    lines = [
        "=" * 60,
        f"📊 PHASE 1 PREPROCESSING - LIVE DASHBOARD",
        "-" * 60,
        f"Total Slides:    {n_total}",
        f"Already Finished: {n_skipped + n_done}  (+{n_done} this session)",
        f"Remaining:       {n_todo - n_done}",
        f"Parallelism:     {num_workers} processes",
        "-" * 60,
        f"Progress: [{bar}] {percent:6.1f}% ",
        f"Speed:    {n_done/max(elapsed_time,1):.2f} slide/s | ETA: {eta_str}",
        "=" * 60,
    ]
    
    if redraw:
        # Remonter de N lignes pour reecrire proprement
        sys.stdout.write(f"\033[{len(lines)}A")
    
    for line in lines:
        sys.stdout.write(f"\r{line}\033[K\n")
    sys.stdout.flush()


def main() -> None:
    """Point d'entree du preprocessing WSI."""

    import time
    args = parse_args()
    slides = discover_slides(args)
    config = build_config(args)
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Statistiques de depart
    skipped = []
    to_process = []
    for slide_path in slides:
        slide_output_csv = config.processed_data_dir / slide_path.stem / "patch_registry.csv"
        if slide_output_csv.exists():
            skipped.append(slide_path)
        else:
            to_process.append(slide_path)

    n_total = len(slides)
    n_skipped = len(skipped)
    n_todo = len(to_process)

    print_stats(n_total, n_skipped, 0, n_todo, args.num_workers)

    if not to_process:
        print("✅ All slides are already processed.")
        return

    if args.num_workers > 1 and n_todo > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import logging
        
        logging.getLogger("prostate_mil").setLevel(logging.WARNING)
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(preprocess_slide, slide_path, config) for slide_path in to_process]
            
            results = []
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                # On redessine tout d'un coup sans tqdm
                print_stats(n_total, n_skipped, len(results), n_todo, args.num_workers, 
                            elapsed_time=time.time()-start_time, redraw=True)
            total_patches = sum(results)
    else:
        results = []
        start_time = time.time()
        for slide_path in to_process:
            res = preprocess_slide(slide_path=slide_path, config=config)
            results.append(res)
            print_stats(n_total, n_skipped, len(results), n_todo, args.num_workers, 
                        elapsed_time=time.time()-start_time, redraw=True)
        total_patches = sum(results)


    print("-" * 60)
    print(f"🎉 Done! Extracted metadata for {total_patches} patches across {n_todo} new slides.")
    print(f"Registries saved in: {config.processed_data_dir}")
    print("=" * 60)



if __name__ == "__main__":
    main()
