# Food Classification

Multi-class food image classification using transfer learning.

## Overview

Two datasets:
- **VippStar** — ~8,668 images, 42 classes, 5-fold CV + holdout val/test
- **Unified** — three sources (VippStar + MedGR + ODIN), 249 classes, 3-fold CV

Four backbones compared: ResNet50, EfficientNet-B3, MobileNetV3-Large, DenseNet121 — all ImageNet pretrained, backbone frozen except the last block.

Tracks top-1, top-3, and top-5 accuracy per epoch. Built as a structured PyTorch codebase (no notebooks).

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
