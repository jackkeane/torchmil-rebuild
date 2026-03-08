# torchmil Rebuild Plan

**Paper:** [A PyTorch-based library for deep Multiple Instance Learning](https://arxiv.org/abs/2509.08129)
**Original:** https://torchmil.readthedocs.io | https://huggingface.co/torchmil
**Started:** 2026-03-08

---

## Overview

Rebuild the `torchmil` library — a modular PyTorch framework for building, training, and evaluating deep MIL models. MIL is a weakly supervised paradigm where labels are assigned to "bags" of instances, not individual instances (e.g., a whole-slide pathology image is a bag, patches are instances).

---

## Phase 1 — Core Data Layer (`torchmil.data`)
**Goal:** Standardized bag representation + batching

- [ ] Define bag representation using `TensorDict`:
  - Instance features tensor `[N_i, D]`
  - Bag-level label
  - Optional: adjacency matrix `[N_i, N_i]` for graph-based models
  - Optional: instance-level labels
- [ ] Implement custom `collate_fn` for variable-length bag batching:
  - Pad instances to max bag size in batch
  - Generate attention masks
- [ ] Unit tests: verify padding, masking, single/multi-bag collation

**Deps:** `torch`, `tensordict`

---

## Phase 2 — Dataset Layer (`torchmil.datasets`)
**Goal:** Standardized dataset loading + at least 1 benchmark

- [ ] `ProcessedMILDataset` base class — efficient loading of pre-extracted features
- [ ] Implement `Camelyon16MIL` dataset class:
  - Download from HuggingFace (`torchmil` org)
  - Accept `root`, `features` (e.g., `"UNI"`, `"ResNet50-BT"`) params
  - Expose `data_dim` attribute
- [ ] Storage format spec (directory layout for features + labels)
- [ ] Optional: algorithmic unit test datasets (Raff & Holt, 2023) for quick iteration

**Deps:** `huggingface_hub`, `datasets`

---

## Phase 3 — Building Blocks (`torchmil.nn`)
**Goal:** Reusable MIL components as PyTorch modules

- [ ] **Attention pooling** — gated attention mechanism (Ilse et al., 2018)
- [ ] **Transformer encoder** — multi-head self-attention for instance interaction
- [ ] **Graph convolution layers** — message passing on instance adjacency graphs
- [ ] **Aggregation operators** — mean/max/attention-weighted pooling
- [ ] **Classification head** — bag-level MLP classifier

---

## Phase 4 — Models (`torchmil.models`) — Start with Top 5
**Goal:** Implement core MIL models, all extending `MILModel` base class

### MILModel base class
- [ ] Unified interface: `__init__(in_shape, criterion)`, `forward(bag)`, `predict(bag)`
- [ ] Standard `state_dict()` save/load support

### Models (priority order)
1. [ ] **ABMIL** (Ilse 2018) — attention-based, the classic baseline
2. [ ] **CLAM** (Lu 2021) — attention + data-efficient, widely used
3. [ ] **TransMIL** (Shao 2021) — transformer-based
4. [ ] **DSMIL** (Li 2021) — dual-stream + contrastive
5. [ ] **DTFDMIL** (Zhang 2022) — double-tier feature distillation

### Later (Phase 4b)
6. [ ] PatchGCN (Chen 2021) — graph-based
7. [ ] DeepGraphSurv (Li 2018)
8. [ ] GTP (Zheng 2022) — graph-transformer
9. [ ] SETMIL (Zhao 2022)
10. [ ] IIBMIL (Ren 2023)
11. [ ] CAMIL (Fourkioti 2024)
12. [ ] SmABMIL (Castro-Macías 2024)
13. [ ] TransformerABMIL (Castro-Macías 2024)
14. [ ] SmTransformerABMIL (Castro-Macías 2024)

---

## Phase 5 — Training & Evaluation (`torchmil.utils`)
**Goal:** Trainer class + metrics

- [ ] `Trainer` class:
  - Accept model, optimizer, device
  - `train(dataloader, epochs)` method
  - Validation loop with early stopping
  - Logging (loss, metrics per epoch)
- [ ] Metrics:
  - Accuracy
  - AUROC
  - F1 score
  - Composite "Performance" = mean(ACC, F1, AUROC)
- [ ] 5-fold cross-validation utility

---

## Phase 6 — Reproduce CAMELYON16 Benchmark
**Goal:** Match paper results on CAMELYON16

- [ ] Download CAMELYON16 pre-extracted features (ResNet50-BT)
- [ ] Train ABMIL, CLAM, TransMIL with: batch_size=1, Adam, lr=1e-4, 50 epochs
- [ ] Compare ACC/AUROC/F1 against Table 1 in paper
- [ ] Document results in `results/camelyon16.md`

**Paper baselines to match:**
| Model | ACC | AUROC | F1 |
|-------|-----|-------|----|
| ABMIL | 0.907 | 0.937 | 0.872 |
| CLAM | 0.907 | 0.933 | 0.867 |
| TransMIL | 0.915 | 0.947 | 0.885 |

---

## Phase 7 — Documentation & Polish
- [ ] README with quickstart
- [ ] API docs (docstrings)
- [ ] Tutorial notebook: end-to-end CAMELYON16 classification
- [ ] `pyproject.toml` packaging

---

## Project Structure

```
torchmil-rebuild/
├── BUILD_PLAN.md
├── pyproject.toml
├── README.md
├── torchmil/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bag.py          # TensorDict bag representation
│   │   └── collate.py      # Custom collate_fn
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py          # ProcessedMILDataset
│   │   └── camelyon16.py    # CAMELYON16 dataset
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── attention.py     # Gated attention
│   │   ├── transformer.py   # MIL transformer
│   │   ├── graph.py         # Graph conv layers
│   │   └── pooling.py       # Aggregation operators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # MILModel base class
│   │   ├── abmil.py
│   │   ├── clam.py
│   │   ├── transmil.py
│   │   ├── dsmil.py
│   │   └── dtfdmil.py
│   └── utils/
│       ├── __init__.py
│       ├── trainer.py       # Trainer class
│       └── metrics.py       # ACC, AUROC, F1
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_trainer.py
├── notebooks/
│   └── camelyon16_tutorial.ipynb
└── results/
    └── camelyon16.md
```

---

## Tech Stack
- Python 3.12 (conda py312)
- PyTorch
- TensorDict (from torchrl)
- HuggingFace Datasets
- scikit-learn (metrics)
- RTX 4090 for training

---

## Timeline Estimate
| Phase | Effort |
|-------|--------|
| 1. Data layer | 1 day |
| 2. Datasets | 1 day |
| 3. NN building blocks | 2 days |
| 4. Models (top 5) | 3-4 days |
| 5. Trainer + eval | 1 day |
| 6. Benchmark repro | 2 days |
| 7. Docs | 1 day |
| **Total** | **~10-12 days** |
