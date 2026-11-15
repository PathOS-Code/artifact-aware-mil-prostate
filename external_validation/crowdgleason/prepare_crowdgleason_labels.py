"""Prepare CrowdGleason labels for external slide-level MIL evaluation.

CrowdGleason provides patch-level labels in Annotations/*.csv. The MIL model
evaluated in this repository predicts slide-level ISUP grades, so this script
groups test patches by WSI identifier and derives a deterministic slide label
from the consensus patch labels. CrowdGleason is external-only here: these
labels are for evaluation, never training or hyperparameter tuning.

Patch ground-truth code used by CrowdGleason:
  0 -> benign / non-cancer
  1 -> Gleason pattern 3
  2 -> Gleason pattern 4
  3 -> Gleason pattern 5

Slide Gleason -> ISUP mapping:
  benign/no cancer -> ISUP 0
  3+3 -> ISUP 1
  3+4 -> ISUP 2
  4+3 -> ISUP 3
  4+4, 3+5, 5+3 -> ISUP 4
  4+5, 5+4, 5+5 -> ISUP 5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PATCH_LABEL_TO_PATTERN = {
    0: 0,
    1: 3,
    2: 4,
    3: 5,
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CrowdGleason external-eval labels and patch manifest."
    )
    parser.add_argument(
        "--crowdgleason-dir",
        type=Path,
        default=Path("data/raw/crowdgleason-dataset"),
        help="Dataset root containing Annotations/ and Patches or NormalizedPatches.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train", "val"],
        help="CrowdGleason split to prepare. External evaluation should use test.",
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=None,
        help=(
            "Patch directory. Defaults to <crowdgleason-dir>/NormalizedPatches/<split> "
            "when present, otherwise <crowdgleason-dir>/Patches/<split>."
        ),
    )
    parser.add_argument(
        "--out-labels",
        type=Path,
        default=Path("external_validation/labels/crowdgleason_labels.csv"),
        help="Output slide-level labels CSV.",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("external_validation/labels/crowdgleason_patch_manifest.csv"),
        help="Output patch-level manifest CSV.",
    )
    parser.add_argument(
        "--label-column",
        default="auto",
        help=(
            "Patch label column to use. 'auto' uses 'ground truth' when present, "
            "otherwise MACE, then MV, then DS, then GLAD."
        ),
    )
    return parser.parse_args()


def parse_slide_id(patch_id: str) -> str:
    match = re.match(r"^(?P<slide>.+)_row_\d+_column_\d+$", patch_id)
    if not match:
        raise ValueError(f"Cannot parse slide_id from patch filename: {patch_id}")
    return match.group("slide")


def parse_coords(patch_id: str) -> tuple[int, int]:
    match = re.search(r"_row_(?P<row>\d+)_column_(?P<col>\d+)$", patch_id)
    if not match:
        return 0, 0
    return int(match.group("col")), int(match.group("row"))


def choose_patch_root(dataset_root: Path, split: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    normalized = dataset_root / "NormalizedPatches" / split
    if normalized.exists():
        return normalized
    return dataset_root / "Patches" / split


def slide_label_from_patch_labels(labels: list[int]) -> tuple[str, int]:
    patterns = [PATCH_LABEL_TO_PATTERN[int(label)] for label in labels]
    cancer_patterns = [pattern for pattern in patterns if pattern > 0]
    if not cancer_patterns:
        return "benign", 0

    counts = Counter(cancer_patterns)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    primary = ordered[0][0]
    secondary = ordered[1][0] if len(ordered) > 1 else primary
    gleason = f"{primary}+{secondary}"
    return gleason, GLEASON_TO_ISUP.get((primary, secondary), 0)


def resolve_label_column(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(
                f"Requested label column '{requested}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        return requested

    for candidate in ["ground truth", "MACE", "MV", "DS", "GLAD"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not auto-detect a label column. Expected one of: "
        "ground truth, MACE, MV, DS, GLAD."
    )


def main() -> None:
    args = parse_args()
    annotations_csv = args.crowdgleason_dir / "Annotations" / f"{args.split}.csv"
    patch_root = choose_patch_root(args.crowdgleason_dir, args.split, args.patch_root)

    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {annotations_csv}")
    if not patch_root.exists():
        raise FileNotFoundError(f"Patch directory not found: {patch_root}")

    df = pd.read_csv(annotations_csv)
    label_column = resolve_label_column(df, args.label_column)
    required = {"Patch filename", label_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required annotation columns: {sorted(missing)}")

    manifest_rows: list[dict] = []
    missing_patches: list[Path] = []
    for _, row in df.iterrows():
        patch_id = str(row["Patch filename"])
        patch_path = patch_root / f"{patch_id}.jpg"
        if not patch_path.exists():
            missing_patches.append(patch_path)

        x, y = parse_coords(patch_id)
        patch_label = int(row[label_column])
        if patch_label not in PATCH_LABEL_TO_PATTERN:
            raise ValueError(
                f"Unexpected patch label {patch_label!r} in column '{label_column}' "
                f"for patch {patch_id}. Expected labels: {sorted(PATCH_LABEL_TO_PATTERN)}"
            )
        manifest_rows.append(
            {
                "slide_id": parse_slide_id(patch_id),
                "patch_id": patch_id,
                "patch_path": str(patch_path),
                "coord_x": x,
                "coord_y": y,
                "patch_ground_truth": patch_label,
                "patch_gleason_pattern": PATCH_LABEL_TO_PATTERN[patch_label],
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    if missing_patches:
        print(f"WARNING: {len(missing_patches)} patch files referenced by labels are missing.")
        for path in missing_patches[:10]:
            print(f"  missing: {path}")
        if len(missing_patches) > 10:
            print(f"  ... and {len(missing_patches) - 10} more")

    label_rows: list[dict] = []
    for slide_id, group in manifest.groupby("slide_id", sort=True):
        gleason_raw, isup_grade = slide_label_from_patch_labels(
            group["patch_ground_truth"].astype(int).tolist()
        )
        counts = Counter(group["patch_gleason_pattern"].astype(int).tolist())
        first_patch_path = str(group["patch_path"].iloc[0])
        label_rows.append(
            {
                "image_id": slide_id,
                "slide_id": slide_id,
                "patch_path": first_patch_path,
                "gleason_raw": gleason_raw,
                "isup_grade": isup_grade,
                "n_patches": int(len(group)),
                "n_benign_patches": int(counts.get(0, 0)),
                "n_gleason3_patches": int(counts.get(3, 0)),
                "n_gleason4_patches": int(counts.get(4, 0)),
                "n_gleason5_patches": int(counts.get(5, 0)),
                "patch_manifest": str(args.out_manifest),
            }
        )

    labels = pd.DataFrame(label_rows).sort_values("slide_id")

    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out_manifest, index=False)
    labels.to_csv(args.out_labels, index=False)

    summary = {
        "dataset_root": str(args.crowdgleason_dir),
        "split": args.split,
        "annotations_csv": str(annotations_csv),
        "patch_root": str(patch_root),
        "n_annotation_rows": int(len(df)),
        "n_manifest_rows": int(len(manifest)),
        "n_unique_slides": int(labels["slide_id"].nunique()),
        "n_missing_patches": int(len(missing_patches)),
        "label_column": label_column,
        "patch_ground_truth_counts": {
            str(k): int(v) for k, v in df[label_column].value_counts().sort_index().items()
        },
        "isup_counts": {
            str(k): int(v) for k, v in labels["isup_grade"].value_counts().sort_index().items()
        },
        "gleason_raw_counts": {
            str(k): int(v) for k, v in labels["gleason_raw"].value_counts().sort_index().items()
        },
    }
    summary_path = args.out_labels.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("CrowdGleason label preparation complete")
    print(f"  Labels:   {args.out_labels}")
    print(f"  Manifest: {args.out_manifest}")
    print(f"  Summary:  {summary_path}")
    print(f"  Label column: {label_column}")
    print(f"  Rows:     {len(manifest)} patches -> {len(labels)} slides")
    print("  Patch ground-truth counts:")
    for key, value in summary["patch_ground_truth_counts"].items():
        print(f"    {key}: {value}")
    print("  Slide ISUP counts:")
    for key, value in summary["isup_counts"].items():
        print(f"    ISUP {key}: {value}")


if __name__ == "__main__":
    main()
