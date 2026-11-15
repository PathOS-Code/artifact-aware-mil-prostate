"""Prépare les labels SICAPv2 au format compatible avec notre pipeline.

Lit wsi_labels.xlsx (Gleason primary/secondary) et convertit en ISUP grades.
Sauvegarde en CSV avec colonnes: image_id, isup_grade
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GLEASON_TO_ISUP = {
    (3, 3): 1,
    (3, 4): 2,
    (4, 3): 3,
    (4, 4): 4,
    (3, 5): 4,
    (5, 3): 4,
    (4, 5): 5,
    (5, 4): 5,
    (5, 5): 5,
}


def gleason_to_isup(primary: int, secondary: int) -> int:
    """Convertit un score Gleason (primary, secondary) en grade ISUP (0-5)."""
    if primary < 3 and secondary < 3:
        return 0  # Benign
    return GLEASON_TO_ISUP.get((primary, secondary), 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertir les labels SICAPv2 Gleason → ISUP.")
    parser.add_argument(
        "--sicapv2-dir", type=Path, required=True,
        help="Racine du dataset SICAPv2 (contient wsi_labels.xlsx)",
    )
    args = parser.parse_args()

    xlsx = args.sicapv2_dir / "wsi_labels.xlsx"
    if not xlsx.exists():
        print(f"❌ wsi_labels.xlsx introuvable dans {args.sicapv2_dir}")
        return

    wsi = pd.read_excel(xlsx)
    print(f"📝 Lu {len(wsi)} slides depuis {xlsx.name}")

    # Convertir Gleason → ISUP
    wsi["isup_grade"] = wsi.apply(
        lambda r: gleason_to_isup(int(r["Gleason_primary"]), int(r["Gleason_secondary"])),
        axis=1,
    )

    # Renommer pour compatibilité avec SlideEmbeddingDataset
    out_df = wsi[["slide_id", "isup_grade"]].rename(columns={"slide_id": "image_id"})

    # Afficher la distribution
    print("\n📊 Distribution ISUP:")
    for grade in range(6):
        count = (out_df["isup_grade"] == grade).sum()
        print(f"  ISUP {grade}: {count} slides")
    print(f"  Total:  {len(out_df)} slides")

    # Sauvegarder
    out_csv = args.sicapv2_dir / "sicapv2_labels.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ Labels sauvegardés dans {out_csv}")


if __name__ == "__main__":
    main()
