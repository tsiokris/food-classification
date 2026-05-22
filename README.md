# Food Classification

Multi-class food image classification using transfer learning. Built as an end-to-end ML engineering project — not a notebook, but a structured Python codebase.

## Overview

- ~8,668 images across 42 food classes (including multi-food combinations)
- ResNet50 backbone pretrained on ImageNet, with partial fine-tuning
- 5-fold cross-validation splits + holdout val/test sets
- Custom PyTorch `Dataset`, `DataLoader`, training loop, and model — all from scratch

## Architecture

| Component | Details |
|---|---|
| Backbone | ResNet50 (ImageNet pretrained) |
| Frozen layers | All except `layer4` and `fc` |
| Classification head | `Linear(2048 → 42)` |
| Optimizer | Adam with per-layer learning rates (`layer4`: 1e-4, `fc`: 1e-3) |
| Loss | CrossEntropyLoss |

## Project Structure

```
src/
  data/
    preprocessing.py   # builds classification CSV from raw annotations
    dataset.py         # FoodDataset (PyTorch Dataset)
    transforms.py      # train/val image transform pipelines
    dataloader.py      # get_dataloader() utility
  models/
    model.py           # FoodClassifier (ResNet50 + custom head)
  training/
    train.py           # train() and validate() loops
main.py                # wires everything together and saves weights
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.10+
- PyTorch 2.3.1
- torchvision 0.18.1
- pandas 2.2.2
- Pillow 10.3.0
