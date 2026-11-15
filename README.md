# Aggregation, ordinal training and artifact gating in MIL for prostate biopsy grading

Publicly available code (PolyForm Noncommercial), data splits and result
files for the paper *Aggregation, Ordinal
Training, and Artifact Gating in Multiple Instance Learning for Prostate
Biopsy Grading* (Boufenar, Lounis, Ounis, Belhaddad; submitted).

The pipeline predicts the ISUP grade group (0–5) of a prostate biopsy
whole-slide image from frozen foundation-model patch embeddings:

1. tissue segmentation and 256 × 256 patch registries (`scripts/preprocessing/`)
2. frozen Virchow2 or HistoEncoder embeddings (`scripts/feature_extraction/`)
3. an optional norm–variance artifact gate that flags outlier patch tokens and
   excludes them from attention (`src/prostate_mil/models/mil/gating.py`)
4. ABMIL or a TransMIL-style aggregator, trained with focal or CORAL loss
   (`src/prostate_mil/models/mil/`, `src/prostate_mil/training/mil/`)
5. seed and fold ensembles, paired bootstrap CIs, and direct transfer to
   SICAPv2 and CrowdGleason (`scripts/evaluation/`, `external_validation/`)

This is research code. It is not a medical device and has not been validated
for clinical use.

## Main results

From `results/`; see the paper for the full analysis and its limitations.

| Setting | QWK |
| --- | --- |
| PANDA held-out test (1,061 slides), TransMIL-style focal five-seed ensemble, full robust gate | 0.9134 (95% CI 0.8856–0.9368) |
| PANDA five-fold CV, TransMIL-style CORAL, full robust gate (out-of-fold, optimistic) | 0.9227 |
| SICAPv2 (155 biopsies), CORAL fold ensemble, no fine-tuning | 0.8772 (95% CI 0.803–0.933) |
| CrowdGleason (232 slide identifiers, surrogate labels), CORAL fold ensemble | 0.8080 (95% CI 0.733–0.871) |

The PANDA test ensemble was the best of three ensembles, selected post hoc.
None of the paired gated-versus-ungated QWK intervals excluded zero. All
intervals treat slides as independent and ignore patient clustering.

## Repository layout

```text
src/prostate_mil/        library: WSI reading, patching, backbones, gate, MIL models, training
scripts/                 command-line entry points (preprocessing → evaluation)
external_validation/     SICAPv2 and CrowdGleason label preparation and feature extraction
splits/                  the fixed PANDA train/val/test split used in the paper
results/                 result files, run configs and training logs behind the paper's tables
data/                    PANDA label file and data access instructions
docs/                    record of the gate threshold calibration
tests/                   unit and integration tests
```

## Installation

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/PathOS-Code/artifact-aware-mil-prostate.git
cd artifact-aware-mil-prostate
uv sync                      # GPU: PyTorch 2.6.0 with CUDA 12.4 wheels
```

CPU only (enough for the tests, the derived analyses and evaluating released
weights on precomputed embeddings):

```bash
uv venv --python 3.12
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install h5py matplotlib numpy openpyxl openslide-bin openslide-python opencv-python \
  pandas pillow scikit-learn seaborn timm torchstain tqdm pytest
```

In a CPU environment use `uv run --no-sync ...` (or activate `.venv` and call
`python` directly) so that uv does not replace the CPU wheels with the locked
CUDA ones. Scripts add `src/` to the import path themselves; no package
install is needed.

On Windows, set `PYTHONUTF8=1` before redirecting script output to a file;
the training scripts print Unicode symbols.

`uv.lock` pins the versions used for the paper (PyTorch 2.6.0+cu124, timm
1.0.26, NumPy 2.4.4, scikit-learn 1.8.0).

## Data

See [`data/README.md`](data/README.md). PANDA, SICAPv2 and CrowdGleason are
public; Virchow2 weights require Hugging Face access approval.

## Reproducing the paper

See [`REPRODUCE.md`](REPRODUCE.md) for every command, from patch extraction to
the external bootstrap. Analyses that only need the stored result files run in
seconds on CPU:

```bash
uv run python scripts/evaluation/ci_from_confusion_matrices.py
```

## Pretrained weights

The MIL heads of the reported ensembles (optimizer state removed, about
160 MB per TransMIL-style member) are available at: TODO (Zenodo DOI).
They are released under CC BY-NC-SA 4.0 because they were trained on PANDA.
They take Virchow2 embeddings as input, so using them requires access to
Virchow2 under its own license.

## Tests

```bash
uv run pytest -m "not network"     # skips tests that download HistoEncoder weights
```

## License

The code is publicly available under the
[PolyForm Noncommercial License 1.0.0](LICENSE): free for research,
education and other noncommercial use, not for commercial use. It is
source-available, not open source in the OSI sense. For commercial licensing,
contact the corresponding author.

| Material | License |
| --- | --- |
| Code (`src/`, `scripts/`, `external_validation/*.py`, `tests/`) | PolyForm Noncommercial 1.0.0 |
| Model weights; PANDA-derived files (`data/panda_labels.csv`, `splits/`, `results/`) | CC BY-NC-SA 4.0, following PANDA |
| CrowdGleason-derived labels (`external_validation/labels/`) | CC BY 4.0, following CrowdGleason |
| PANDA, SICAPv2, CrowdGleason images; Virchow2 and HistoEncoder weights | Their providers' licenses, not redistributed here |

Using the pipeline with Virchow2 or PANDA is in any case limited to
noncommercial use by their own terms.

## Citation

See [`CITATION.cff`](CITATION.cff). TODO: add the journal reference once
published.

## Contributors

- Akram Samir Ounis, University of Algiers 1 Benyoucef Benkhedda
- Chawki Ahmed Ramzi Belhaddad, University of Algiers 1 Benyoucef Benkhedda
- Dr. Khaled Lounis, pathologist, University of Algiers 1 Benyoucef Benkhedda
- Prof. Chaouki Boufenar, École Nationale Supérieure d'Intelligence Artificielle (ENSIA) and University of Algiers 1 Benyoucef Benkhedda
- Prof. Mohammed Al-Sarem, Taibah University, Saudi Arabia
