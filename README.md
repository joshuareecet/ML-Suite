# ML Foundations

PyTorch image classification experiments across progressively deeper architectures and multiple datasets.

## Setup

```bash
python -m venv .env && source .env/bin/activate
pip install -r requirements.txt
python train.py
```

Dataset downloads automatically. Normalisation stats are computed once and cached in `data/dataset_config.json`.

## Configuration

```python
# train.py
MODEL        = Res50
DATASET      = datasets.CIFAR10
DATASET_NAME = "CIFAR10"
```

`in_channels`, `num_classes`, and `imgsz` are derived from the dataset automatically.

## Models

| Class | Architecture |
|---|---|
| `SimpleMLP` | Flatten → 512 → 512 → out |
| `SimpleCNN` | 2× Conv+BN+ReLU+MaxPool → FC(256) → out |
| `StridedCNN` | 3× Conv+BN+ReLU (stride-2) → FC(256) → out |
| `MiniResNet` | 3 residual blocks (32→64→128) → FC(256) → out |
| `Res50` | 4 bottleneck stages (64→256→512→1024→2048) → AdaptiveAvgPool → out |

## Hyperparameters

| | |
|---|---|
| Epochs | 50 |
| Batch size | 64 |
| LR | 1e-3 |
| LR schedule | Linear decay to 0.5× over 30 epochs |
| Optimizer | AdamW |

Best checkpoint saved to `models/` on each validation loss improvement.
