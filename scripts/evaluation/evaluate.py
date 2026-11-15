"""Script CLI pour l'évaluation d'un modèle entrainé et génération de matrice de confusion."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prostate_mil.models.mil.model import CustomMILModel
from prostate_mil.training.mil.dataset import SlideEmbeddingDataset
from prostate_mil.training.mil.losses import coral_predict
from prostate_mil.training.mil.splits import fixed_split_indices
from prostate_mil.utils.io import find_checkpoint, load_model_state
from prostate_mil.config.defaults import GatingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluer un modèle et générer la matrice de confusion.")
    parser.add_argument("--exp-dir", type=Path, required=True,
                        help="Chemin vers le dossier de l'expérience (contenant config.json et best_model.safetensors ou best_model.pt)")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Dossier contenant les embeddings (.pt)")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="Chemin vers le CSV des labels (ISUP grade)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Ratio de validation utilisé pendant l'entrainement (pour retrouver le même split)")
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="Ratio de test (pour un split 3-way train/val/test). 0 = pas de test set.")
    parser.add_argument("--split-csv", type=Path, default=None,
                        help="Fixed split file (slide_id,split). Overrides --val-ratio/--test-ratio.")
    parser.add_argument("--eval-set", type=str, choices=["val", "test"], default="test",
                        help="Partition à évaluer: 'val' ou 'test' (default: test si test-ratio > 0, sinon val)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for results (default: same as --exp-dir)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir if args.output_dir else args.exp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (args.exp_dir / "config.json").exists() or find_checkpoint(args.exp_dir) is None:
        print(f"❌ Erreur: config.json ou poids (best_model.safetensors / best_model.pt) introuvables dans {args.exp_dir}")
        return

    # 1. Charger la config de l'expérience
    with open(args.exp_dir / "config.json", "r") as f:
        config = json.load(f)

    print(f"📝 Loading experiment: {config.get('experiment_name', 'Unknown')}")

    dev = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )

    # 2. Reconstruire le modèle
    # Récupérer l'aggregator (default à "abmil" si manquant dans les vieux configs)
    aggregator = config.get("aggregator", "abmil") if "transmil_layers" not in config or config["transmil_layers"] == "N/A" else "transmil"
    if "transmil_layers" in config and config["transmil_layers"] != "N/A":
        aggregator = "transmil"
    transmil_layers = config.get("transmil_layers", 2)
    if transmil_layers == "N/A":
        transmil_layers = 2
        
    _mode_aliases = {"v1": "standard", "v2": "robust"}
    gating_mode = config.get("gating_mode", "robust")
    gating_mode = _mode_aliases.get(gating_mode, gating_mode)
    use_gating = config.get("gating", True)
    
    # 3. Load checkpoint first, infer input_dim from the saved weights
    gc = GatingConfig()
    state_dict = load_model_state(args.exp_dir)
    
    # Infer input_dim from the attention layer weight shape
    if "attention.attention_v.weight" in state_dict:
        detected_dim = state_dict["attention.attention_v.weight"].shape[1]
    elif "attention.pooling.attention_v.weight" in state_dict:  # TransMIL wraps ABMIL
        detected_dim = state_dict["attention.pooling.attention_v.weight"].shape[1]
    else:
        detected_dim = gc.input_dim
    print(f"🔍 Model input_dim (from checkpoint): {detected_dim}")
    
    ordinal = config.get("loss_type", "focal") == "coral"
    model = CustomMILModel(
        input_dim=detected_dim,
        attention_dim=gc.attention_dim,
        num_classes=6,
        variance_threshold=config.get("variance_threshold", gc.variance_threshold),
        use_gating=use_gating,
        gating_mode=gating_mode if gating_mode != "N/A" else "robust",
        dropout=config.get("dropout", 0.25),
        aggregator=aggregator,
        transmil_layers=transmil_layers,
        ordinal=ordinal,
    ).to(dev)

    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Modèle chargé")

    # 4. Préparer le Dataset (même split que l'entrainement)
    full_dataset = SlideEmbeddingDataset(
        embeddings_dir=args.embeddings_dir,
        labels_csv=args.labels_csv,
    )

    n_total = len(full_dataset)

    # On utilise la MEME SEED que dans train.py pour reconstruire le split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()

    if args.split_csv is not None or args.test_ratio > 0:
        # 3-way split: train / val / test (same logic as train.py)
        train_idx, val_idx, test_idx = fixed_split_indices(
            full_dataset, args.val_ratio, args.test_ratio, split_csv=args.split_csv,
        )
        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

        eval_set = args.eval_set
        if eval_set == "test":
            eval_idx = test_idx
            eval_name = "test"
        else:
            eval_idx = val_idx
            eval_name = "validation"
        dataset = Subset(full_dataset, eval_idx)
        print(f"📦 Split 3-way: train={n_train}, val={n_val}, test={n_test}")
        print(f"📦 Evaluating on {eval_name} set ({len(eval_idx)} slides)...")
    else:
        # 2-way split: train / val
        n_val = int(n_total * args.val_ratio)
        n_train = n_total - n_val
        val_idx = indices[n_train:]
        dataset = Subset(full_dataset, val_idx)
        eval_name = "validation"
        print(f"📦 Evaluating on validation set ({n_val} slides)...")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 5. Boucle d'inférence
    all_true = []
    all_pred = []

    print("🚀 Début de l'inférence...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            embeddings = batch["embeddings"].squeeze(0).to(dev)
            variances = batch["spatial_variances"].squeeze(0).to(dev)
            label = batch["label"].item()

            output = model(embeddings, variances)
            logits = output["logits"].unsqueeze(0)
            pred = coral_predict(logits).item() if ordinal else logits.argmax(dim=1).item()

            all_true.append(label)
            all_pred.append(pred)
            
            if (i+1) % 500 == 0:
                print(f"  ... traité {i+1} / {len(dataset)}")

    # 6. Matrice de confusion
    cm = confusion_matrix(all_true, all_pred, labels=range(6))
    
    # 7. Affichage et sauvegarde
    plt.figure(figsize=(8, 6))
    
    # ISUP Grades class names
    classes = [f"ISUP {i}" for i in range(6)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar=False, annot_kws={"size": 12})
                
    plt.title(f"Confusion Matrix ({config.get('experiment_name', 'Model')})", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True ISUP Grade', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted ISUP Grade', fontsize=12, fontweight='bold')
    
    # Calcul des accuracies par classe dans les annotations (optionnel, mais utile)
    # class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    
    plt.tight_layout()
    out_img = output_dir / "confusion_matrix.png"
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    print(f"\n📊 Matrice de confusion sauvegardée dans : {out_img}")
    
    report_file = output_dir / "classification_report.txt"
    with open(report_file, "w") as f:
        # Also compute QWK
        from prostate_mil.training.mil.engine import _quadratic_weighted_kappa
        qwk = _quadratic_weighted_kappa(all_true, all_pred)
        
        f.write(f"Evaluating: {args.exp_dir.name}\n")
        f.write(f"{eval_name.capitalize()} QWK: {qwk:.4f}\n\n")
        f.write(classification_report(
            all_true, all_pred,
            target_names=[f"ISUP {i}" for i in range(6)],
            labels=list(range(6)),
            zero_division=0,
        ))
        
    print(f"📄 Rapport de classification sauvegardé dans : {report_file}")
    
    # Print it to the console as well
    with open(report_file, "r") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
