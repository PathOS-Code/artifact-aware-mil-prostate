import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from pathlib import Path

def plot_gating_analysis(embeddings_dir="outputs/embeddings_virchow2", output_dir="outputs/experiments/gating_analysis_virchow2", num_slides=100, z_score_factor=2.0, variance_threshold=122.0):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pt_files = list(Path(embeddings_dir).glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {embeddings_dir}")
        return
        
    random.seed(42)
    sample_files = random.sample(pt_files, min(num_slides, len(pt_files)))
    
    all_norms = []
    all_vars = []
    all_masks = []
    
    print(f"Loading {len(sample_files)} slides for Virchow2 gating analysis...")
    
    for f in sample_files:
        data = torch.load(f, map_location="cpu", weights_only=True)
        emb = data["embeddings"]
        var = data["spatial_variances"]
        
        # Calculate norm
        norms = torch.linalg.vector_norm(emb, dim=-1)
        
        # Calculate V2 statistics (Median/MAD) PER SLIDE just like the CustomGatingLayer
        norm_median = norms.median()
        abs_deviation = torch.abs(norms - norm_median)
        mad = abs_deviation.median() * 1.4826
        mad = torch.clamp(mad, min=1e-8)
        
        robust_z = abs_deviation / mad
        outlier_mask = robust_z > z_score_factor
        
        low_var_mask = var < variance_threshold
        artifact_mask = outlier_mask & low_var_mask
        
        all_norms.append(norms.numpy())
        all_vars.append(var.numpy())
        all_masks.append(artifact_mask.numpy())
        
    all_norms = np.concatenate(all_norms)
    all_vars = np.concatenate(all_vars)
    all_masks = np.concatenate(all_masks)
    
    total_patches = len(all_norms)
    num_artifacts = all_masks.sum()
    artifact_ratio = num_artifacts / total_patches * 100
    
    print(f"Total patches: {total_patches:,}")
    print(f"Identified artifacts: {num_artifacts:,} ({artifact_ratio:.2f}%)")
    
    # 1. Histogram
    plt.figure(figsize=(10, 6))
    
    valid_norms = all_norms[~all_masks]
    artifact_norms = all_norms[all_masks]
    
    # Calculate global median and MAD just for visualizing the overall cutoff
    g_median = np.median(all_norms)
    g_mad = np.median(np.abs(all_norms - g_median)) * 1.4826
    lower_cutoff = g_median - z_score_factor * g_mad
    upper_cutoff = g_median + z_score_factor * g_mad
    
    plt.hist(valid_norms, bins=100, alpha=0.7, color='blue', label='Valid Patches')
    if len(artifact_norms) > 0:
        plt.hist(artifact_norms, bins=50, alpha=0.7, color='red', label='Flagged Artifacts')
    
    plt.axvline(g_median, color='black', linestyle='-', label=f'Global Median: {g_median:.2f}')
    plt.axvline(upper_cutoff, color='red', linestyle='--', label=f'Upper Cutoff: {upper_cutoff:.2f}')
    if lower_cutoff > 0:
        plt.axvline(lower_cutoff, color='red', linestyle='--', label=f'Lower Cutoff: {lower_cutoff:.2f}')
        
    plt.title('Virchow2 (1280d) - Token L2 Norm Distribution')
    plt.xlabel('L2 Norm')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histogram_norms_virchow2.png", dpi=300)
    plt.close()
    
    # 2. Scatter Plot
    plt.figure(figsize=(10, 6))
    
    # Randomly subsample for scatter plot to avoid overplotting (max 50k points)
    if total_patches > 50000:
        idx = np.random.choice(total_patches, 50000, replace=False)
        sub_norms = all_norms[idx]
        sub_vars = all_vars[idx]
        sub_masks = all_masks[idx]
    else:
        sub_norms = all_norms
        sub_vars = all_vars
        sub_masks = all_masks
        
    plt.scatter(sub_vars[~sub_masks], sub_norms[~sub_masks], alpha=0.3, s=5, c='blue', label='Valid')
    if sub_masks.sum() > 0:
        plt.scatter(sub_vars[sub_masks], sub_norms[sub_masks], alpha=0.7, s=15, c='red', label='Artifacts')
        
    plt.axvline(variance_threshold, color='green', linestyle='--', label=f'Var Cutoff: {variance_threshold}')
    plt.axhline(upper_cutoff, color='red', linestyle='--', label='Norm Cutoff (Approx)')
    
    plt.title('Virchow2 - Spatial Variance vs L2 Norm')
    plt.xlabel('Spatial Variance (Laplacian)')
    plt.ylabel('L2 Norm (Virchow2 1280d)')
    # Use log scale for variance as it often spans orders of magnitude
    if np.min(sub_vars) > 0:
        plt.xscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_variance_vs_norm_virchow2.png", dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_gating_analysis()
