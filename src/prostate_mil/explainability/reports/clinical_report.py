"""Rapport clinique structure pour une lame analysee."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np



@dataclass
class ClinicalReport:
    """Resume clinique complet d'une lame."""

    slide_name: str
    predicted_isup: int
    class_probabilities: list[float]
    n_patches_total: int
    n_artifacts_filtered: int
    heatmap_path: str | None
    overlay_path: str | None
    # Gleason analysis (populated when mask is available)
    gleason_score: str | None = None
    gleason_primary: int | None = None
    gleason_secondary: int | None = None
    isup_from_mask: int | None = None
    cancer_percentage: float | None = None
    gleason_area_breakdown: dict | None = None
    gleason_heatmap_path: str | None = None
    combined_heatmap_path: str | None = None


def save_report_json(report: ClinicalReport, output_path: Path) -> Path:
    """Serialise le rapport clinique en JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "slide_name": report.slide_name,
        "predicted_isup": report.predicted_isup,
        "class_probabilities": report.class_probabilities,
        "n_patches_total": report.n_patches_total,
        "n_artifacts_filtered": report.n_artifacts_filtered,
        "heatmap_path": report.heatmap_path,
        "overlay_path": report.overlay_path,
    }

    # Gleason fields (only if mask was provided)
    if report.gleason_score is not None:
        data["gleason_analysis"] = {
            "gleason_score": report.gleason_score,
            "primary_pattern": report.gleason_primary,
            "secondary_pattern": report.gleason_secondary,
            "isup_from_mask": report.isup_from_mask,
            "cancer_percentage": report.cancer_percentage,
            "area_breakdown": report.gleason_area_breakdown,
            "model_vs_mask_agreement": report.predicted_isup == report.isup_from_mask,
        }
        if report.gleason_heatmap_path:
            data["gleason_heatmap_path"] = report.gleason_heatmap_path
        if report.combined_heatmap_path:
            data["combined_heatmap_path"] = report.combined_heatmap_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path
