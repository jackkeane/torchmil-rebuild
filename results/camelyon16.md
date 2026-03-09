# CAMELYON16 Benchmark Reproduction (Phase 6)

## How to run

From repository root:

```bash
python scripts/run_camelyon16_benchmark.py \
  --data-root ./data \
  --features UNI \
  --batch-size 1 \
  --lr 1e-4 \
  --epochs 50 \
  --models abmil clam transmil \
  --results-dir results
```

If data is not already local, you can request auto-download from Hugging Face:

```bash
python scripts/run_camelyon16_benchmark.py --data-root ./data --features UNI --download
```

The script expects:

- `data/camelyon16/<features>/manifest.csv`
- feature files referenced by `features_path` in the manifest
- `split` values matching `train` and `test` (configurable via flags)

## Expected benchmark settings

- Models: ABMIL, CLAM, TransMIL
- Optimizer: Adam
- Learning rate: `1e-4`
- Epochs: `50`
- Batch size: `1`
- Metrics: ACC, AUROC, F1, composite performance (`(ACC + AUROC + F1) / 3`)

## Outputs

Per-model files:

- `results/camelyon16_abmil.json`, `results/camelyon16_abmil.csv`
- `results/camelyon16_clam.json`, `results/camelyon16_clam.csv`
- `results/camelyon16_transmil.json`, `results/camelyon16_transmil.csv`

Summary files:

- `results/camelyon16_summary.json`
- `results/camelyon16_summary.csv`

## Current status

- ✅ Benchmark implemented and validated with smoke tests.
- ✅ Real benchmark run completed on local RTX 4090 (UNI features).

### Current reproduced metrics (local run)

| Model | ACC | AUROC | F1 | Performance |
|---|---:|---:|---:|---:|
| ABMIL | 0.96899 | 0.99745 | 0.95745 | 0.97463 |
| CLAM | 0.96124 | 0.99898 | 0.94624 | 0.96882 |
| TransMIL | 0.96899 | 0.99235 | 0.95833 | 0.97322 |

(Full machine-readable outputs are in `camelyon16_*.json/.csv` and `camelyon16_summary.*`.)
