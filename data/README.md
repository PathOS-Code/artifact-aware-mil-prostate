# Data

No image data is distributed with this repository. Download each dataset from
its provider and accept its license.

| Dataset | Source | License | Used for |
| --- | --- | --- | --- |
| PANDA | [Kaggle: prostate-cancer-grade-assessment](https://www.kaggle.com/c/prostate-cancer-grade-assessment) | CC BY-NC-SA 4.0 | development (train / val / test) |
| SICAPv2 | Mendeley Data (Silva-Rodríguez et al., 2020) | see provider | external test |
| CrowdGleason | Zenodo (López-Pérez et al., 2024) | CC BY 4.0 | external test |

`panda_labels.csv` is the PANDA `train.csv` label file (10,616 slides:
`image_id, data_provider, isup_grade, gleason_score`). Training and evaluation
scripts take it with `--labels-csv data/panda_labels.csv`.

Expected local layout (all git-ignored):

```text
data/
  panda/train_images/*.tiff        PANDA WSIs
  registries/<slide_id>/patch_registry.csv   written by scripts/preprocessing/preprocess.py
outputs/
  embeddings_virchow2/<slide_id>.pt          written by scripts/feature_extraction/extract_features.py
```

Any other location works; pass it on the command line.

## Encoder weights

- **Virchow2**: gated on Hugging Face (`paige-ai/Virchow2`). Request access,
  log in with `huggingface-cli login`, and the weights are fetched through
  timm on first use. Check the model card for its license terms.
- **HistoEncoder** (`prostate_small`): downloaded automatically from the
  authors' release URL in `src/prostate_mil/models/backbones/backbone.py`.

Embeddings computed with these encoders are not redistributed here.
