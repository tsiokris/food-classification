# Food Classification — Subtask Report

## 1. Problem Statement

The Food Reader is an end-to-end pipeline that estimates the nutritional content of a meal from an image. It is composed of three sequential models:

1. **Segmentation model** — identifies and localises individual food items within the image.
2. **Classification model** — identifies what food each segmented region contains.
3. **Weight regressor** — estimates the weight (in grams) of each identified food item.

The final output of the pipeline is a per-item weight estimate, which feeds directly into nutritional calculations. The classification step is a prerequisite for the regressor: the food class must be known before a class-specific weight model can be applied.

This report covers the classification subtask. Given a (potentially segmented) food image, the classifier predicts which food class the image belongs to, from a fixed set of ~42 classes. The classes include both single-food items and multi-food combinations (e.g. a plate containing both rice and chicken is treated as a distinct combined class).

---

## 2. Dataset Description

The dataset was collected internally and consists of approximately **8,668 food images** spanning **42 classes**.

### Annotation structure

Two annotation files are used:

- `metadata.csv` — contains the image path and up to three food labels per image, along with per-label weight annotations (used by the downstream regressor, not the classifier).
- `final_split.csv` — contains the image path and its assigned data split.

### Multi-food handling

Images may contain more than one food item. When multiple labels are present, they are concatenated with an underscore to form a single compound class label (e.g. `rice_chicken`). This flattens the problem into standard single-label classification, at the cost of treating each unique combination as a separate class.

### Data splits

The dataset is partitioned into seven named splits:

| Split | Role |
|---|---|
| `fold_0` – `fold_4` | Training (5-fold cross-validation) |
| `holdout_val` | Validation during development |
| `holdout_test` | Final held-out evaluation |

In the current training run, all five folds are used together as the training set, with `holdout_val` and `holdout_test` combined as the evaluation set.

---

## 3. Methodology

### 3.1 Architecture

Four ImageNet-pretrained backbones were evaluated. In each case the original classification head is replaced with a new linear layer mapping to 42 output classes, the full backbone is frozen, and only the last block of convolutional layers plus the new head are trained.

**ResNet50**

```
ResNet50 (pretrained, ImageNet1K_V1)
  └── conv1, bn1, relu, maxpool   — frozen
  └── layer1                      — frozen
  └── layer2                      — frozen
  └── layer3                      — frozen
  └── layer4                      — fine-tuned  (lr = 1e-4)
  └── fc: Linear(2048 → 42)       — trained from scratch (lr = 1e-3)
```

**EfficientNet-B3**

```
EfficientNet-B3 (pretrained, ImageNet1K_V1)
  └── features[0–5]               — frozen  (stem + first 6 MBConv stages)
  └── features[6]                 — fine-tuned  (lr = 1e-4)
  └── features[7]                 — fine-tuned  (lr = 1e-4)
  └── features[8]                 — fine-tuned  (lr = 1e-4)  (final projection conv)
  └── classifier[1]: Linear(1536 → 42) — trained from scratch (lr = 1e-3)
```

**MobileNet-V3-Large**

```
MobileNet-V3-Large (pretrained, ImageNet1K_V1)
  └── features[0–12]              — frozen  (stem + first InvertedResidual blocks)
  └── features[13]                — fine-tuned  (lr = 1e-4)
  └── features[14]                — fine-tuned  (lr = 1e-4)
  └── features[15]                — fine-tuned  (lr = 1e-4)
  └── features[16]                — fine-tuned  (lr = 1e-4)  (final projection conv)
  └── classifier[3]: Linear(1280 → 42) — trained from scratch (lr = 1e-3)
```

**DenseNet121**

```
DenseNet121 (pretrained, ImageNet1K_V1)
  └── features.conv0 / norm0      — frozen
  └── features.denseblock1–3      — frozen
  └── features.transition1–3      — frozen
  └── features.denseblock4        — fine-tuned  (lr = 1e-4)
  └── classifier: Linear(1024 → 42) — trained from scratch (lr = 1e-3)
```

### 3.2 Training strategy

The same strategy is applied to all four backbones:

1. Load ImageNet-pretrained weights.
2. Freeze all backbone parameters.
3. Unfreeze the last convolutional block (see §3.1 per-model details).
4. Replace the classification head with a new `Linear` layer initialised randomly.
5. Train with two Adam parameter groups — a slow rate for the unfrozen block and a fast rate for the new head.

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Loss | Cross-entropy |
| Last block lr | 1e-4 |
| Head lr | 1e-3 |
| Batch size | 32 |
| Epochs | 10 |
| Input size | 224 × 224 |

### 3.3 Data augmentation and preprocessing

| Stage | Transforms |
|---|---|
| Training | `RandomResizedCrop(224)`, `RandomHorizontalFlip`, `ToTensor`, ImageNet normalisation |
| Validation / Test | `Resize(224, 224)`, `ToTensor`, ImageNet normalisation |

Augmentation is applied only at training time to improve generalisation. ImageNet mean and standard deviation normalisation is applied to match the distribution the backbone was pretrained on.

---

## 4. Design Choices and Reasoning

**Transfer learning (ResNet50 pretrained on ImageNet)**  
Training a deep CNN from scratch on ~8,668 images would be very likely to overfit. Starting from ImageNet weights gives the model strong general visual representations (edges, textures, shapes) for free. ResNet50 offers a well-established accuracy–compute tradeoff and is a reliable baseline for image classification tasks.

**Freezing most of the backbone**  
The early and middle layers of a ResNet learn low-level and mid-level features (edges, gradients, textures) that transfer well across image domains. Freezing them prevents the model from destroying these representations on a small dataset and significantly reduces the number of trainable parameters, which helps avoid overfitting.

**Unfreezing `layer4`**  
The final residual block encodes the highest-level semantic representations, which tend to be the most task-specific. Allowing it to fine-tune lets the model adapt its abstract feature extraction to food-specific semantics while still benefiting from the pretrained initialisation.

**Differential learning rates**  
`layer4` already contains useful representations, so it is fine-tuned with a small learning rate to avoid disrupting them. The `fc` head is randomly initialised and has no prior structure, so it benefits from a larger learning rate that allows faster convergence.

**Multi-food as a single class**  
Treating multi-food combinations as compound labels (e.g. `rice_chicken`) keeps the problem formulation simple: it remains standard single-label multi-class classification with Cross-Entropy loss, with no changes needed to the model or training loop.

---

## 5. Results

Four backbones were trained for 10 epochs on the VippStar dataset (~8,668 images, 42 classes) using the strategy described above. All models were evaluated on the combined `holdout_val` + `holdout_test` split.

### 5.1 Summary table

| Model | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Acc (epoch) |
|---|---|---|---|---|---|
| ResNet50 | 0.1473 | 95.24% | 0.0486 | 98.73% | **99.19%** (ep. 8) |
| DenseNet121 | 0.1412 | 95.52% | 0.0527 | 98.27% | 98.92% (ep. 6) |
| EfficientNet-B3 | 0.1787 | 94.30% | 0.0320 | **99.08%** | **99.08%** (ep. 10) |
| MobileNet-V3-Large | 0.1364 | **95.86%** | 0.0360 | 98.85% | 98.92% (ep. 7) |

All four models converge to above **98% validation accuracy** within 10 epochs, demonstrating that the transfer learning setup is well-suited for this dataset size.

### 5.2 Observations

- **EfficientNet-B3** achieves the lowest final validation loss (0.032) and best final val accuracy (99.08%), and is still improving at epoch 10 — it may benefit from further training.
- **ResNet50** peaks earliest (epoch 8, 99.19%) but shows slight overfitting towards the end.
- **MobileNet-V3-Large** has the best final training accuracy (95.86%) and converges cleanly, making it a strong candidate for a lightweight deployment model.
- **DenseNet121** performs slightly below the others in final val accuracy, but converges stably.

### 5.3 Training curves

Training and validation loss/accuracy curves for all four models are saved at `results/training_curves.png`.

### 5.4 Class distribution

The class distribution plot (`results/class_distribution.png`) shows the 42 classes sorted by frequency. The dataset has moderate imbalance, with single-food classes generally having more samples than multi-food compound classes.

---

## 6. Limitations

- **Compound label explosion:** Each unique food combination becomes a distinct class. With a small dataset, some combinations may have very few training examples, making them hard to learn reliably.
- **No class imbalance handling:** Some classes (especially rare multi-food combinations) will be underrepresented. No re-weighting or oversampling is applied.
- **Flat multi-label treatment:** Treating combinations as a single class means the model cannot partially predict components of a multi-food image. A true multi-label approach would be more expressive.
- **Limited evaluation metrics:** Training currently tracks only loss and top-1 accuracy. There is no per-class breakdown, confusion matrix, or reporting of performance on rare classes.
- **Input assumes clean crops:** The classifier receives an image and predicts a class, but does not explicitly receive a segmentation mask. Performance will depend on how cleanly the upstream segmentation model isolates individual food items.

---

## 7. Future Work

- **Per-class metrics and confusion matrix:** Understanding where the model fails (e.g. which classes are confused with each other) is important for targeted improvement.
- **Class imbalance mitigation:** Weighted cross-entropy loss or oversampling of minority classes could improve performance on rare combinations.
- **Multi-label reformulation:** Replacing the compound-label scheme with true multi-label classification (e.g. binary cross-entropy per class) would be more principled for multi-food images.
- **Experiment tracking:** Integrating a tool such as Weights & Biases would allow systematic comparison of hyperparameter configurations and training runs.
- **Hyperparameter search:** Learning rates, the number of unfrozen layers, batch size, and augmentation policy are all candidates for tuning.
- **Inference script:** A standalone inference module that accepts an image path and returns a predicted class and confidence score would make it straightforward to integrate the classifier into the full Food Reader pipeline.
