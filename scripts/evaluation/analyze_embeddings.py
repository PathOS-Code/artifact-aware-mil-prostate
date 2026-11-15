"""Pre-training analysis: check if gating will actually help.

Answers:
1. Does HistoEncoder produce artifact tokens (high L2 norm outliers)?
2. What's the L2 norm distribution?
3. What's the spatial variance distribution?
4. What % of tokens would the current threshold flag?
5. Are the flagged tokens actually artifacts or real tissue?
"""

import torch
import numpy as np
from pathlib import Path
import random

random.seed(42)

embeddings_dir = Path("outputs/embeddings")
pts = list(embeddings_dir.glob("*.pt"))

# Sample 200 random slides for analysis
sample_pts = random.sample(pts, min(200, len(pts)))

all_norms = []
all_variances = []
per_slide_stats = []

print(f"Analyzing {len(sample_pts)} slides...")

for pt_path in sample_pts:
    data = torch.load(pt_path, weights_only=False)
    emb = data["embeddings"]  # [N, 384]
    var = data["spatial_variances"]  # [N]

    norms = torch.linalg.vector_norm(emb, ord=2, dim=1)  # [N]

    mu = norms.mean().item()
    sigma = norms.std().item()
    cutoff = mu + 2.0 * sigma

    # Dual condition: high norm AND low variance
    high_norm = norms > cutoff
    low_var_15 = var < 15.0
    low_var_50 = var < 50.0
    low_var_100 = var < 100.0

    artifacts_15 = (high_norm & low_var_15).sum().item()
    artifacts_50 = (high_norm & low_var_50).sum().item()
    artifacts_100 = (high_norm & low_var_100).sum().item()
    n = len(norms)

    per_slide_stats.append({
        "slide": pt_path.stem,
        "n_patches": n,
        "norm_mean": mu,
        "norm_std": sigma,
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
        "norm_cutoff": cutoff,
        "high_norm_count": high_norm.sum().item(),
        "artifacts_var15": artifacts_15,
        "artifacts_var50": artifacts_50,
        "artifacts_var100": artifacts_100,
        "var_mean": var.mean().item(),
        "var_std": var.std().item(),
        "var_min": var.min().item(),
        "var_max": var.max().item(),
    })

    all_norms.extend(norms.numpy().tolist())
    all_variances.extend(var.numpy().tolist())

all_norms = np.array(all_norms)
all_variances = np.array(all_variances)

print(f"\n{'='*60}")
print(f"EMBEDDING L2 NORM ANALYSIS ({len(all_norms):,} tokens from {len(sample_pts)} slides)")
print(f"{'='*60}")
print(f"  Mean:     {all_norms.mean():.4f}")
print(f"  Std:      {all_norms.std():.4f}")
print(f"  Min:      {all_norms.min():.4f}")
print(f"  Max:      {all_norms.max():.4f}")
print(f"  Median:   {np.median(all_norms):.4f}")
print(f"  P1:       {np.percentile(all_norms, 1):.4f}")
print(f"  P5:       {np.percentile(all_norms, 5):.4f}")
print(f"  P95:      {np.percentile(all_norms, 95):.4f}")
print(f"  P99:      {np.percentile(all_norms, 99):.4f}")

# Check for outliers
global_mu = all_norms.mean()
global_sigma = all_norms.std()
global_cutoff = global_mu + 2 * global_sigma
outliers = (all_norms > global_cutoff).sum()
print(f"\n  Global cutoff (mu+2*sigma): {global_cutoff:.4f}")
print(f"  Tokens above cutoff: {outliers:,} / {len(all_norms):,} ({outliers/len(all_norms)*100:.2f}%)")

print(f"\n{'='*60}")
print(f"SPATIAL VARIANCE ANALYSIS")
print(f"{'='*60}")
print(f"  Mean:     {all_variances.mean():.4f}")
print(f"  Std:      {all_variances.std():.4f}")
print(f"  Min:      {all_variances.min():.4f}")
print(f"  Max:      {all_variances.max():.4f}")
print(f"  Median:   {np.median(all_variances):.4f}")
print(f"  P1:       {np.percentile(all_variances, 1):.4f}")
print(f"  P5:       {np.percentile(all_variances, 5):.4f}")
print(f"  P10:      {np.percentile(all_variances, 10):.4f}")
print(f"  P25:      {np.percentile(all_variances, 25):.4f}")

# Artifact detection at different thresholds
print(f"\n{'='*60}")
print(f"ARTIFACT DETECTION (per-slide mu+2*sigma + variance threshold)")
print(f"{'='*60}")
for threshold in [15.0, 50.0, 100.0, 200.0, 500.0]:
    total_flagged = 0
    total_tokens = 0
    for s in per_slide_stats:
        total_tokens += s["n_patches"]
        # Recompute with this threshold
    # Use the precomputed values
    if threshold == 15.0:
        flagged = sum(s["artifacts_var15"] for s in per_slide_stats)
    elif threshold == 50.0:
        flagged = sum(s["artifacts_var50"] for s in per_slide_stats)
    elif threshold == 100.0:
        flagged = sum(s["artifacts_var100"] for s in per_slide_stats)
    else:
        flagged = -1

    total_tokens = sum(s["n_patches"] for s in per_slide_stats)
    if flagged >= 0:
        print(f"  variance_threshold={threshold:6.1f}  →  {flagged:,} / {total_tokens:,} flagged ({flagged/total_tokens*100:.3f}%)")

print(f"\n{'='*60}")
print(f"CONCLUSION")
print(f"{'='*60}")

# Does HistoEncoder produce outliers?
ratio_99 = all_norms[all_norms > np.percentile(all_norms, 99)].mean() / all_norms.mean()
print(f"  P99/mean norm ratio: {ratio_99:.2f}x")
if ratio_99 > 1.5:
    print(f"  ✅ HistoEncoder DOES produce high-norm outlier tokens (ratio {ratio_99:.2f}x)")
    print(f"     → Gating layer WILL have something to filter")
else:
    print(f"  ⚠️  HistoEncoder produces mild outliers (ratio {ratio_99:.2f}x)")
    print(f"     → Gating effect may be small")

# Recommend variance threshold
median_var = np.median(all_variances)
p10_var = np.percentile(all_variances, 10)
print(f"\n  Variance median: {median_var:.1f}, P10: {p10_var:.1f}")
print(f"  Recommended variance_threshold: {p10_var:.0f}")
print(f"  (Catches bottom 10% lowest-variance patches as candidates)")
