"""Export trained runs as release weights (safetensors) and check them.

For each run directory (``best_model.pt`` + ``config.json``) this writes
``OUT_ROOT/<name>/``:

- ``best_model.safetensors``: the model state_dict only (fp32 tensors, no
  optimizer state, no pickled objects)
- ``config.json``: the run's recorded configuration, copied unchanged
- ``release_info.json``: best epoch and validation QWK from the checkpoint,
  SHA-256 of the weights, and the conversion check result

The conversion is checked by building the model from ``config.json`` twice,
once with the original state_dict and once with the safetensors file, and
comparing logits on random embedding bags. A format conversion does not depend
on the input data, so random bags are sufficient; the export fails if any
logit differs.

Usage:
  uv run python scripts/training/export_weights.py \\
    --run RUN_DIR NAME [--run RUN_DIR NAME ...] --out-root release/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "evaluation"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch
from safetensors.torch import load_file, save_file

from ensemble_eval import build_model_from_config

N_CHECK_BAGS = 3


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _max_logit_diff(config: dict, original: dict, converted: dict) -> float:
    device = torch.device("cpu")
    model_a = build_model_from_config(config, original, device)
    model_b = build_model_from_config(config, converted, device)
    input_dim = next(v for k, v in original.items() if k.endswith("attention_v.weight")).shape[1]

    generator = torch.Generator().manual_seed(0)
    worst = 0.0
    with torch.no_grad():
        for n_patches in (64, 257, 1000)[:N_CHECK_BAGS]:
            embeddings = torch.randn(n_patches, input_dim, generator=generator)
            variances = torch.rand(n_patches, generator=generator) * 800.0
            a = model_a(embeddings, variances)["logits"]
            b = model_b(embeddings, variances)["logits"]
            worst = max(worst, float((a - b).abs().max()))
    return worst


def export_run(run_dir: Path, name: str, out_root: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    state = {k: v.detach().to(torch.float32).contiguous() for k, v in ckpt["model_state_dict"].items()}

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "best_model.safetensors"
    save_file(state, str(weights_path))
    shutil.copy2(run_dir / "config.json", out_dir / "config.json")

    diff = _max_logit_diff(config, ckpt["model_state_dict"], load_file(str(weights_path)))
    if diff != 0.0:
        raise RuntimeError(f"{name}: converted weights change the logits (max abs diff {diff})")

    info = {
        "name": name,
        "best_epoch": ckpt.get("epoch"),
        "best_val_qwk": ckpt.get("best_qwk"),
        "n_parameters": int(sum(v.numel() for v in state.values())),
        "weights_sha256": _sha256(weights_path),
        "conversion_check": {
            "random_bags": N_CHECK_BAGS,
            "max_abs_logit_diff": diff,
        },
    }
    (out_dir / "release_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", nargs=2, action="append", metavar=("RUN_DIR", "NAME"), required=True,
                    help="Source run directory and release name (NAME may contain '/').")
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    for run_dir, name in args.run:
        info = export_run(Path(run_dir), name, args.out_root)
        size_mb = (args.out_root / name / "best_model.safetensors").stat().st_size / 1e6
        print(f"{name}: {size_mb:.0f} MB, epoch {info['best_epoch']}, "
              f"val QWK {info['best_val_qwk']:.4f}, logit diff {info['conversion_check']['max_abs_logit_diff']}")


if __name__ == "__main__":
    main()
