# Results

Output files of the runs reported in the paper, copied unchanged from the
original experiment directories and renamed. JSON files hold the full record
(per-member QWK, confusion matrix, and for CrowdGleason per-slide
predictions); TXT files hold the scikit-learn classification report.

Naming: `<encoder>_<aggregator>_<loss>_<gate>_<protocol>`, where gate is
`nogate`, `maskgate` (mask-only gate) or `fullgate` (full robust gate), and
protocol is `5seed` (five initialization seeds on the fixed split) or `5fold`
(stratified five-fold cross-validation). The mask-only and full robust gate
variants implement the same model (paper, Section 4); their differences are
replicate differences.

The `exp_dirs` and `inputs` fields inside the JSON files are the paths of the
original experiment directories, kept as recorded.

## PANDA (`panda/`)

Fixed split (`splits/panda_fixed_split.csv`, 1,061 test slides).

| File | Ensemble | Test QWK |
| --- | --- | --- |
| `virchow2_transmil_focal_nogate_5seed` | TransMIL-style, focal, no gate, 5 seeds | 0.9063 |
| `virchow2_transmil_focal_maskgate_5seed` | TransMIL-style, focal, mask-only gate, 5 seeds | 0.9104 |
| `virchow2_transmil_focal_fullgate_5seed` | TransMIL-style, focal, full robust gate, 5 seeds | 0.9134 |
| `virchow2_abmil_focal_maskgate_5seed` | ABMIL, focal, mask-only gate, 5 seeds | 0.8919 |

`bootstrap_ci_transmil_focal_5seed.json` (and the `_summary.csv` view) holds
the paired bootstrap (10,000 resamples) of the three TransMIL-style focal
ensembles: point estimates, 95% CIs and pairwise deltas.

Five-fold cross-validation (CORAL loss, out-of-fold QWK over 10,614 slides):

| File | Gate | Out-of-fold QWK |
| --- | --- | --- |
| `virchow2_transmil_coral_maskgate_5fold_summary.json` | mask-only | 0.9226 |
| `virchow2_transmil_coral_fullgate_5fold_summary.json` | full robust | 0.9227 |

`single_runs/` holds the classification reports of the single fixed-split runs
(Virchow2 ABMIL and TransMIL-style, HistoEncoder 4-layer TransMIL-style control).

## SICAPv2 (`sicapv2/`)

155 biopsies, direct transfer without fine-tuning.

| File | QWK |
| --- | --- |
| `virchow2_transmil_focal_nogate_5seed` | 0.8664 |
| `virchow2_transmil_focal_maskgate_5seed` | 0.8633 |
| `virchow2_transmil_focal_fullgate_5seed` | 0.8482 |
| `virchow2_transmil_coral_maskgate_5fold` | 0.8760 |
| `virchow2_transmil_coral_fullgate_5fold` | 0.8772 |

`single_runs/` holds SICAPv2 classification reports of single models.

## CrowdGleason (`crowdgleason/`)

Official curated test partition, 232 slide identifiers, surrogate slide labels
derived from patch labels (`external_validation/crowdgleason/prepare_crowdgleason_labels.py`).

| File | QWK |
| --- | --- |
| `virchow2_transmil_focal_nogate_5seed` | 0.8076 |
| `virchow2_transmil_focal_maskgate_5seed` | 0.8028 |
| `virchow2_transmil_focal_fullgate_5seed` | 0.8076 |
| `virchow2_transmil_coral_maskgate_5fold` | 0.8109 |
| `virchow2_transmil_coral_fullgate_5fold` | 0.8080 |

`bootstrap_ci_transmil_focal_5seed.json` holds the paired bootstrap of the
three focal five-seed ensembles (10,000 resamples, seed 2026).

## Run records (`runs/`)

`config.json` and `training_log.json` of every member behind the tables above
(37 runs). Released weights use the same directory names.

## Gate analysis (`gating/`)

L2 norm histogram and norm-versus-variance scatter of Virchow2 patch tokens,
produced by `scripts/evaluation/plot_virchow2_gating.py`.

## Derived analyses

These run on the files above without embeddings or GPU:

- `scripts/evaluation/ci_from_confusion_matrices.py`: bootstrap CIs of the
  ensemble QWK and Clopper–Pearson CIs of sensitivity and specificity at
  GG ≥ 2, from the stored confusion matrices.
- `scripts/evaluation/training_log_divergence.py`: first epoch at which
  same-seed members of the five-seed sets diverge.
- `external_validation/crowdgleason/surrogate_label_recount.py`: checks where
  the CrowdGleason surrogate rule differs from the ISUP needle-biopsy convention.
