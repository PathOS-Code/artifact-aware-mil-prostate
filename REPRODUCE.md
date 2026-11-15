# Reproducing the paper

Commands are written for bash and run from the repository root. Each step
names the files it writes and the matching file under `results/`.

Training is not bit-wise reproducible on GPU even at a fixed seed (paper,
Section 5): expect retrained models to match the reported QWK to about the
second or third decimal, not exactly. The released weights reproduce the
reported ensemble numbers.

Hardware used for the paper: one NVIDIA RTX 3070 (8 GB), 32 GB RAM. One
TransMIL-style training run over the fixed split takes several hours; a
five-fold CORAL run took 15–18 h.

```bash
EMB=outputs/embeddings_virchow2
LABELS=data/panda_labels.csv
SPLIT=splits/panda_fixed_split.csv
```

## 1. Tissue segmentation and patch registries (PANDA)

```bash
uv run python scripts/preprocessing/preprocess.py \
  --slides-dir data/panda/train_images \
  --output-dir data/registries \
  --num-workers 8
```

Writes `data/registries/<slide_id>/patch_registry.csv` (256 × 256 patch
coordinates at level 0, tissue threshold 0.15). No patch images are saved.

## 2. Feature extraction

```bash
uv run python scripts/feature_extraction/extract_features.py \
  --slides-dir data/registries \
  --raw-data-dir data/panda/train_images \
  --backbone-name virchow2 \
  --output-dir $EMB
```

Writes one `<slide_id>.pt` bag per slide (`embeddings`, `spatial_variances`,
`coords`). Use `--backbone-name histoencoder --output-dir outputs/embeddings`
for the HistoEncoder control. Virchow2 requires Hugging Face access (see
`data/README.md`).

## 3. Split

The split used in the paper is `splits/panda_fixed_split.csv`; see
`splits/README.md` for how it was recovered and checked. Pass it to every
fixed-split command with `--split-csv $SPLIT`.

## 4. Training

Gate variants map to these flags:

| Variant | Flags |
| --- | --- |
| No gate | `--no-gating` |
| Mask-only gate | `--gating-mode robust --alpha 1.0` |
| Full robust gate | `--gating-mode robust` (default `--alpha 1e-4`) |

The mask-only runs in the paper were trained with an earlier revision that fed
unscaled tokens to the aggregator; `--alpha 1.0` gives the same computation in
the current code. Because flagged tokens are masked from attention, the two
gated variants are the same model (paper, Section 4).

### 4.1 TransMIL-style, focal loss, five seeds (Table: fixed-split ensembles)

```bash
TRANSMIL="--embeddings-dir $EMB --labels-csv $LABELS --split-csv $SPLIT \
  --backbone virchow2 --aggregator transmil --transmil-layers 2 \
  --lr 1e-5 --weight-decay 0.01 --dropout 0.4 --patch-dropout 0.2 \
  --epochs 50 --accumulation-steps 8 --cosine-lr"

for s in 0 1 2 3 4; do
  uv run python scripts/training/train_mil.py $TRANSMIL --no-gating \
    --seed $s --name virchow2_transmil_focal_nogate_seed$s
  uv run python scripts/training/train_mil.py $TRANSMIL --gating-mode robust --alpha 1.0 \
    --seed $s --name virchow2_transmil_focal_maskgate_seed$s
  uv run python scripts/training/train_mil.py $TRANSMIL --gating-mode robust \
    --seed $s --name virchow2_transmil_focal_fullgate_seed$s
done
```

Runs are written to `outputs/experiments/virchow2/transmil/<name>/`.
Recorded configurations: `results/runs/virchow2_transmil_focal_*_5seed/`.

### 4.2 ABMIL, focal loss, five seeds

```bash
ABMIL="--embeddings-dir $EMB --labels-csv $LABELS --split-csv $SPLIT \
  --backbone virchow2 --aggregator abmil \
  --lr 1e-4 --weight-decay 0.01 --dropout 0.25 --patch-dropout 0.2 \
  --epochs 50 --accumulation-steps 8"

for s in 0 1 2 3 4; do
  uv run python scripts/training/train_mil.py $ABMIL --gating-mode robust --alpha 1.0 \
    --seed $s --name virchow2_abmil_focal_maskgate_seed$s
done
```

Single fixed-split ABMIL runs: same flags with `--seed 0`, with and without
`--no-gating`.

### 4.3 TransMIL-style, CORAL loss, five-fold cross-validation

```bash
CV="--embeddings-dir $EMB --labels-csv $LABELS \
  --backbone virchow2 --aggregator transmil --transmil-layers 2 --loss coral \
  --lr 1e-5 --weight-decay 0.01 --dropout 0.4 --patch-dropout 0.2 \
  --epochs 50 --accumulation-steps 8 --cosine-lr --kfold 5 --seed 0"

uv run python scripts/training/train_mil.py $CV --gating-mode robust \
  --name virchow2_transmil_coral_fullgate_5fold
uv run python scripts/training/train_mil.py $CV --gating-mode robust --alpha 1.0 --early-stopping 10 \
  --name virchow2_transmil_coral_maskgate_5fold
```

The mask-only CV run in the paper used early stopping with patience 10; the
full robust gate run did not. Fold membership cannot be restored exactly
(`splits/README.md`), so out-of-fold QWK will differ slightly from
`results/panda/*_5fold_summary.json`.

### 4.4 HistoEncoder control

```bash
uv run python scripts/training/train_mil.py \
  --embeddings-dir outputs/embeddings --labels-csv $LABELS --split-csv $SPLIT \
  --backbone histoencoder --aggregator transmil --transmil-layers 4 \
  --lr 1e-4 --weight-decay 0.01 --dropout 0.25 --patch-dropout 0.2 \
  --epochs 50 --accumulation-steps 8 --cosine-lr \
  --gating-mode robust --alpha 1.0 --seed 0 --name histoencoder_transmil4L_maskgate
```

Add `--no-gating` and change the name for the ungated run.

## 5. Ensemble evaluation on the PANDA test split

```bash
RUNS=outputs/experiments/virchow2/transmil
uv run python scripts/evaluation/ensemble_eval.py \
  --exp-dirs $RUNS/virchow2_transmil_focal_fullgate_seed{0,1,2,3,4} \
  --embeddings-dir $EMB --labels-csv $LABELS --split-csv $SPLIT \
  --out outputs/ensembles/panda/virchow2_transmil_focal_fullgate_5seed.json
```

Repeat for `nogate`, `maskgate` and the ABMIL set. Compare with
`results/panda/`.

Paired bootstrap of the three TransMIL-style ensembles:

```bash
uv run python scripts/evaluation/bootstrap_ci.py \
  --nogate-dirs $RUNS/virchow2_transmil_focal_nogate_seed{0,1,2,3,4} \
  --mask-dirs   $RUNS/virchow2_transmil_focal_maskgate_seed{0,1,2,3,4} \
  --full-dirs   $RUNS/virchow2_transmil_focal_fullgate_seed{0,1,2,3,4} \
  --embeddings-dir $EMB --labels-csv $LABELS --split-csv $SPLIT \
  --n-bootstrap 10000 --out-dir outputs/statistical_tests
```

The resampling seed of the original PANDA bootstrap was not recorded; the
script default is 123, so CI bounds can differ from
`results/panda/bootstrap_ci_transmil_focal_5seed.json` in the fourth decimal.

## 6. External validation

### SICAPv2

```bash
uv run python external_validation/sicapv2/prepare_sicapv2_labels.py --sicapv2-dir data/sicapv2
uv run python external_validation/sicapv2/extract_features_sicapv2.py \
  --images-dir data/sicapv2/images --backbone-name virchow2 \
  --output-dir outputs/external_embeddings/sicapv2/virchow2

uv run python scripts/evaluation/ensemble_eval.py \
  --exp-dirs $RUNS/virchow2_transmil_focal_fullgate_seed{0,1,2,3,4} \
  --embeddings-dir outputs/external_embeddings/sicapv2/virchow2 \
  --labels-csv data/sicapv2/sicapv2_labels.csv \
  --val-ratio 0 --test-ratio 0 \
  --out outputs/ensembles/sicapv2/virchow2_transmil_focal_fullgate_5seed.json
```

With `--val-ratio 0 --test-ratio 0` and no `--split-csv`, every slide in the
labels file is evaluated. For the CV fold ensembles, pass the five
`fold_k` directories as `--exp-dirs`.

### CrowdGleason

```bash
uv run python external_validation/crowdgleason/prepare_crowdgleason_labels.py \
  --crowdgleason-dir data/raw/crowdgleason-dataset --split test
uv run python external_validation/crowdgleason/extract_features_crowdgleason.py

uv run python scripts/evaluation/ensemble_eval.py \
  --exp-dirs $RUNS/virchow2_transmil_focal_fullgate_seed{0,1,2,3,4} \
  --embeddings-dir outputs/external_embeddings/crowdgleason/virchow2 \
  --labels-csv external_validation/labels/crowdgleason_labels.csv \
  --val-ratio 0 --test-ratio 0 \
  --out outputs/ensembles/crowdgleason/virchow2_transmil_focal_fullgate_5seed.json
```

The label and manifest files written by the first command are already in
`external_validation/labels/` (232 slide identifiers, 2,926 patches). Their
patch paths assume the Zenodo archive is unpacked at
`data/raw/crowdgleason-dataset/` and that the normalized patches
(`NormalizedPatches/test/`) were used.

Paired bootstrap of the three focal ensembles:

```bash
uv run python external_validation/bootstrap_external_compare.py \
  --full-json   outputs/ensembles/crowdgleason/virchow2_transmil_focal_fullgate_5seed.json \
  --mask-json   outputs/ensembles/crowdgleason/virchow2_transmil_focal_maskgate_5seed.json \
  --nogate-json outputs/ensembles/crowdgleason/virchow2_transmil_focal_nogate_5seed.json \
  --out-json outputs/statistical_tests/crowdgleason_bootstrap.json
```

## 7. Analyses that need no embeddings

These run on the files in `results/` and reproduce the paper's derived
numbers exactly:

```bash
uv run python scripts/evaluation/ci_from_confusion_matrices.py
uv run python scripts/evaluation/training_log_divergence.py
uv run python external_validation/crowdgleason/surrogate_label_recount.py
```

## 8. Gate calibration and figures

- `scripts/evaluation/analyze_embeddings.py`: L2 norm and Laplacian variance
  distributions over 200 random HistoEncoder bags in `outputs/embeddings`,
  used to set the variance threshold τ = 122 (P10). Record:
  `docs/embedding_analysis.md`.
- `scripts/evaluation/plot_virchow2_gating.py`: norm histogram and
  norm-versus-variance scatter over 100 Virchow2 bags (`results/gating/`).
- `scripts/inference/generate_heatmap.py`: attention heatmap and JSON report
  for one slide:

```bash
uv run python scripts/inference/generate_heatmap.py \
  --slide-path data/panda/train_images/<slide_id>.tiff \
  --embeddings-pt $EMB/<slide_id>.pt \
  --experiment-dir $RUNS/virchow2_transmil_focal_fullgate_seed0 \
  --output-dir outputs/heatmaps
```

## Released weights

To evaluate without training, download the released run directories
(`best_model.pt` + `config.json`, same names as `results/runs/`) and pass them
as `--exp-dirs`. The weights were stripped of optimizer state with
`scripts/training/strip_checkpoint.py`.
