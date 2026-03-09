# torchmil-rebuild

A clean-room rebuild of the `torchmil` ecosystem from the paper: **A PyTorch-based library for deep Multiple Instance Learning** (arXiv:2509.08129).

## What is implemented

- `torchmil.data`: TensorDict bag representation + variable-length collate
- `torchmil.datasets`: `ProcessedMILDataset`, `Camelyon16MIL` (HF download + manifest build)
- `torchmil.nn`: gated attention, transformer encoder, graph conv, pooling, bag classifier
- `torchmil.models`: `MILModel`, `ABMIL`, `CLAM`, `TransMIL`, `DSMIL`, `DTFDMIL`
- `torchmil.utils`: metrics, trainer, 5-fold split utility
- `scripts/run_camelyon16_benchmark.py`: benchmark runner for ABMIL/CLAM/TransMIL

## Quickstart

### 1) Environment

```bash
cd ~/clawd/torchmil-rebuild
~/anaconda3/bin/conda run -n py312 python -m pip install -e .
```

### 2) Run tests

```bash
cd ~/clawd/torchmil-rebuild
PYTHONPATH=. ~/anaconda3/bin/conda run -n py312 python -m pytest tests/ -v
```

### 3) Run CAMELYON16 benchmark (UNI features)

```bash
cd ~/clawd/torchmil-rebuild
PYTHONPATH=. ~/anaconda3/bin/conda run -n py312 python scripts/run_camelyon16_benchmark.py \
  --data-root ./data \
  --features UNI \
  --download \
  --repo-id torchmil/Camelyon16_MIL \
  --epochs 50 \
  --batch-size 1 \
  --lr 1e-4 \
  --models abmil clam transmil \
  --device cuda \
  --results-dir results
```

Results are written to `results/camelyon16_*.json|csv` and `results/camelyon16_summary.*`.

## Minimal API example

```python
from torch.utils.data import DataLoader
from torchmil.datasets import Camelyon16MIL
from torchmil.data import mil_collate_fn
from torchmil.models import ABMIL
from torchmil.utils import Trainer
import torch

dataset = Camelyon16MIL(root="data", features="UNI", split="train", download=True)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=mil_collate_fn)

model = ABMIL(in_shape=dataset.data_dim, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = Trainer(model=model, optimizer=optimizer, device="cuda")

logs = trainer.train(loader, epochs=10)
print(logs[-1])
```

## Project layout

- `torchmil/` core library
- `tests/` unit + smoke tests
- `scripts/` runnable benchmark scripts
- `results/` benchmark outputs/report markdown
- `notebooks/` tutorial notebook
