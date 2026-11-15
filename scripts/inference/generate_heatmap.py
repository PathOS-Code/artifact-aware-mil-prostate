"""Script CLI pour l'inference + heatmap + Gleason pattern analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: inference MIL + heatmap + optional Gleason analysis."
    )
    parser.add_argument(
        "--slide-path", type=Path, required=True,
        help="Path to the WSI (.svs, .tiff) for thumbnail extraction.",
    )
    parser.add_argument(
        "--embeddings-pt", type=Path, required=True,
        help="Phase 2 embeddings .pt file (contains embeddings, coords, variances).",
    )
    parser.add_argument(
        "--experiment-dir", type=Path, required=True,
        help="Experiment directory containing config.json and best_model.pt.",
    )
    parser.add_argument(
        "--mask-path", type=Path, default=None,
        help="Optional: Radboud Gleason mask TIFF. Enables Gleason pattern "
             "heatmap and score derivation.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/heatmaps"),
        help="Output directory.",
    )
    parser.add_argument("--thumbnail-level", type=int, default=2)
    parser.add_argument("--colormap", type=int, default=2, help="OpenCV colormap code (2=JET).")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Optional: Force provider ('radboud' or 'karolinska'). "
             "If None, tries to auto-detect from --labels-csv."
    )
    parser.add_argument(
        "--labels-csv", type=Path, default=Path("data/panda_labels.csv"),
        help="CSV with image_id and data_provider columns (PANDA train.csv format).",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model_from_config(
    config: dict, state_dict: dict, device: torch.device
):
    """Reconstruct model architecture from saved config.json + checkpoint."""
    from prostate_mil.config.defaults import GatingConfig, TrainingConfig
    from prostate_mil.models.mil.model import CustomMILModel

    # Infer input_dim from checkpoint weights (robust to config drift)
    if "attention.attention_v.weight" in state_dict:
        input_dim = state_dict["attention.attention_v.weight"].shape[1]
    elif "attention.pooling.attention_v.weight" in state_dict:
        input_dim = state_dict["attention.pooling.attention_v.weight"].shape[1]
    else:
        input_dim = GatingConfig().input_dim

    # Resolve aggregator
    transmil_layers = config.get("transmil_layers", 2)
    if transmil_layers == "N/A":
        aggregator = "abmil"
        transmil_layers = 2
    else:
        aggregator = "transmil"

    _mode_aliases = {"v1": "standard", "v2": "robust", "N/A": "robust"}
    gating_mode = config.get("gating_mode", "robust")
    gating_mode = _mode_aliases.get(gating_mode, gating_mode)

    gc = GatingConfig()
    ordinal = config.get("loss_type", "focal") == "coral"
    model = CustomMILModel(
        input_dim=input_dim,
        attention_dim=gc.attention_dim,
        num_classes=TrainingConfig().num_classes,
        variance_threshold=config.get("variance_threshold", gc.variance_threshold),
        alpha=gc.alpha,
        use_gating=config.get("gating", True),
        gating_mode=gating_mode,
        dropout=config.get("dropout", 0.25),
        patch_dropout=0.0,  # inference — no stochastic patch drop
        aggregator=aggregator,
        transmil_layers=transmil_layers,
        ordinal=ordinal,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    from prostate_mil.data.wsi.wsi_reader import WSIReader
    from prostate_mil.explainability.reports.clinical_report import (
        ClinicalReport,
        save_report_json,
    )
    from prostate_mil.explainability.attention.heatmap import (
        build_attention_heatmap,
        colorize_heatmap,
    )
    from prostate_mil.explainability.attention.overlay import overlay_heatmap_on_thumbnail

    args = parse_args()
    dev = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load embeddings data ─────────────────────────────────────
    data = torch.load(args.embeddings_pt, weights_only=False)
    embeddings = data["embeddings"]          # [N, D]
    variances = data["spatial_variances"]    # [N]
    coords = data["coords"].numpy()          # [N, 2] — (coord_x, coord_y)

    slide_name = args.embeddings_pt.stem

    # Detect provider
    provider = args.provider
    if provider is None:
        train_csv = args.labels_csv
        if train_csv.exists():
            try:
                df_train = pd.read_csv(train_csv)
                match = df_train[df_train["image_id"] == slide_name]
                if not match.empty:
                    provider = match.iloc[0]["data_provider"]
                else:
                    provider = "radboud" # default
            except Exception:
                provider = "radboud"
        else:
            provider = "radboud"
    
    print(f"Slide: {slide_name} -- Provider: {provider} -- {len(embeddings)} patches")

    # ── 2. Load model from experiment config + checkpoint ───────────
    config_path = args.experiment_dir / "config.json"
    ckpt_path = args.experiment_dir / "best_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {args.experiment_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing best_model.pt in {args.experiment_dir}")

    config = json.loads(config_path.read_text())
    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = build_model_from_config(config, checkpoint["model_state_dict"], dev)

    print(f"Model: {config.get('experiment_name', 'unknown')} "
          f"(gating={config.get('gating')}, aggregator="
          f"{'transmil' if config.get('transmil_layers', 'N/A') != 'N/A' else 'abmil'})")

    # ── 3. Run inference ────────────────────────────────────────────
    with torch.no_grad():
        output = model(embeddings.to(dev), variances.to(dev))

    logits = output["logits"].cpu()
    attention_weights = output["attention_weights"].cpu().numpy()
    artifact_mask = output["artifact_mask"].cpu().numpy()
    predicted_isup = int(torch.argmax(logits).item())
    probs = torch.softmax(logits, dim=0).numpy().tolist()

    n_artifacts = int(artifact_mask.sum())
    print(f"Prediction: ISUP {predicted_isup} -- artifacts: {n_artifacts}/{len(embeddings)}")

    # ── 4. Read WSI thumbnail + build attention heatmap ─────────────
    reader = WSIReader(args.slide_path)
    reader.open()

    level = args.thumbnail_level
    thumbnail_rgb = reader.read_level_rgb(level)
    downsample = reader.get_level_downsample(level)
    reader.close()

    canvas_shape = thumbnail_rgb.shape[:2]

    # Build a DataFrame from the .pt coords for heatmap projection
    patch_df = pd.DataFrame({"coord_x": coords[:, 0], "coord_y": coords[:, 1]})

    heatmap_gray = build_attention_heatmap(
        patch_registry=patch_df,
        attention_scores=attention_weights,
        canvas_shape=canvas_shape,
        patch_size=256,
        downsample_factor=downsample,
    )
    heatmap_bgr = colorize_heatmap(heatmap_gray, colormap=args.colormap)
    overlay_rgb = overlay_heatmap_on_thumbnail(thumbnail_rgb, heatmap_bgr)

    # ── 5. Save attention heatmap outputs ───────────────────────────
    heatmap_path = args.output_dir / f"{slide_name}_heatmap.png"
    overlay_path = args.output_dir / f"{slide_name}_overlay.png"
    report_path = args.output_dir / f"{slide_name}_report.json"

    cv2.imwrite(str(heatmap_path), heatmap_bgr)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    print(f"\nAttention heatmap saved:")
    print(f"  Heatmap:  {heatmap_path.name}")
    print(f"  Overlay:  {overlay_path.name}")

    # ── 6. Gleason analysis (optional — requires mask) ──────────────
    gleason_kwargs = {}

    if args.mask_path is not None:
        if not args.mask_path.exists():
            print(f"\nWarning: mask not found at {args.mask_path}, skipping Gleason analysis.")
        else:
            from prostate_mil.explainability.gleason.gleason import (
                assign_gleason_per_patch,
                build_combined_heatmap,
                build_gleason_heatmap,
                derive_gleason_score,
            )

            print(f"\nRunning Gleason pattern analysis ({len(coords)} patches)...")

            dominant_patterns, area_breakdown = assign_gleason_per_patch(
                mask_path=args.mask_path,
                coords=coords,
                patch_size=256,
                provider=provider,
            )

            gleason_info = derive_gleason_score(area_breakdown, provider=provider)

            print(f"  Gleason Score (from mask): {gleason_info['gleason_score']}")
            print(f"  ISUP from mask: {gleason_info['isup_from_mask']}  |  "
                  f"Model prediction: ISUP {predicted_isup}")
            if gleason_info["cancer_percentage"] > 0:
                print(f"  Cancer area: {gleason_info['cancer_percentage']:.1f}% of tissue")
                for pat, pct in gleason_info.get("pattern_areas", {}).items():
                    print(f"    {pat}: {pct:.1f}%")

            # Build Gleason-colored heatmap
            gleason_heatmap_bgr = build_gleason_heatmap(
                coords=coords,
                dominant_patterns=dominant_patterns,
                canvas_shape=canvas_shape,
                patch_size=256,
                downsample_factor=downsample,
            )
            gleason_overlay = overlay_heatmap_on_thumbnail(
                thumbnail_rgb, gleason_heatmap_bgr, alpha=0.55,
            )

            # Build combined heatmap (Gleason color + attention brightness)
            combined_bgr = build_combined_heatmap(
                coords=coords,
                attention_scores=attention_weights,
                dominant_patterns=dominant_patterns,
                canvas_shape=canvas_shape,
                patch_size=256,
                downsample_factor=downsample,
            )
            combined_overlay = overlay_heatmap_on_thumbnail(
                thumbnail_rgb, combined_bgr, alpha=0.55,
            )

            # Save Gleason outputs
            gleason_path = args.output_dir / f"{slide_name}_gleason.png"
            combined_path = args.output_dir / f"{slide_name}_combined.png"

            cv2.imwrite(str(gleason_path),
                        cv2.cvtColor(gleason_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(combined_path),
                        cv2.cvtColor(combined_overlay, cv2.COLOR_RGB2BGR))

            print(f"\nGleason heatmaps saved:")
            print(f"  Gleason:  {gleason_path.name}  "
                  "(green=benign, yellow=G3, orange=G4, red=G5)")
            print(f"  Combined: {combined_path.name}  "
                  "(Gleason color + attention brightness)")

            gleason_kwargs = dict(
                gleason_score=gleason_info["gleason_score"],
                gleason_primary=gleason_info["primary_pattern"],
                gleason_secondary=gleason_info["secondary_pattern"],
                isup_from_mask=gleason_info["isup_from_mask"],
                cancer_percentage=gleason_info["cancer_percentage"],
                gleason_area_breakdown=area_breakdown,
                gleason_heatmap_path=str(gleason_path),
                combined_heatmap_path=str(combined_path),
            )

    # ── 7. Save clinical report ─────────────────────────────────────
    clinical = ClinicalReport(
        slide_name=slide_name,
        predicted_isup=predicted_isup,
        class_probabilities=probs,
        n_patches_total=len(embeddings),
        n_artifacts_filtered=n_artifacts,
        heatmap_path=str(heatmap_path),
        overlay_path=str(overlay_path),
        **gleason_kwargs,
    )
    save_report_json(clinical, report_path)

    print(f"\n  Report:   {report_path.name}")


if __name__ == "__main__":
    main()
