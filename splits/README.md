# PANDA fixed split

`panda_fixed_split.csv` lists every PANDA development slide with its partition
(`slide_id,split`). Pass it to training and evaluation with
`--split-csv splits/panda_fixed_split.csv`.

| Split | Slides in file | Slides used in the paper |
| --- | --- | --- |
| train | 7,963 | 7,961 |
| val | 1,592 | 1,592 |
| test | 1,061 | 1,061 |

## How the file was built

The original runs did not write the split to disk. `train.py` built it with
`torch.randperm(n, generator=torch.Generator().manual_seed(42))` over the slides
of `data/panda_labels.csv` that had an embedding file, in CSV order, then cut it
75/15/10. Two of the 10,616 slides had no embedding (corrupted source files) and
their identifiers were not recorded.

The split was recovered from the per-slide ISUP labels the original runs cached
for the validation and test sets (1,592 and 1,061 labels, in evaluation order).
Only one placement of the two excluded slides in the label CSV reproduces both
label sequences, position by position. It fixes the validation and test sets
exactly; the two excluded slides are known only to within two short windows of
training slides:

- one of rows 2225–2234 of `data/panda_labels.csv`
  (`378f0ea443c45b7cb1cb8e0cc86a5992` … `37bcf02187f518f188dc8d97038f3e84`)
- one of rows 10055–10063
  (`f327a26eac60aff04d28d21df275b52e` … `f34713c3ba1e433268c056d42b29fef6`)

(Rows are 0-based, header excluded.) Any choice inside those windows gives the
same validation and test sets. All 7,963 candidate training slides are therefore
listed as `train`; slides without an embedding are skipped with a warning when
the split is loaded.

Per-grade counts (ISUP 0–5):

| Split | GG0 | GG1 | GG2 | GG3 | GG4 | GG5 |
| --- | --- | --- | --- | --- | --- | --- |
| train (file) | 2,147 | 2,048 | 993 | 909 | 930 | 936 |
| val | 436 | 382 | 207 | 193 | 193 | 181 |
| test | 309 | 236 | 143 | 140 | 126 | 107 |

Validation and test rows are stored in the original permutation order, so
evaluation scripts, including the bootstrap resampling in
`scripts/evaluation/bootstrap_ci.py`, iterate the test slides in the same order
as the original runs.

## Five-fold cross-validation

The CORAL five-fold experiment uses `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)` over the 10,614 slides with embeddings (`run_kfold_training`
in `src/prostate_mil/training/mil/train.py`). Fold membership depends on which
two slides were excluded, so the original folds cannot be reproduced exactly
and no fold file is provided.
