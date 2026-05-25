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

The classifier is built on a **ResNet50** backbone pretrained on ImageNet. The final fully-connected layer is replaced with a new linear layer mapping the 2048-dimensional feature vector to the 42 output classes.

```
ResNet50 (pretrained)
  └── [conv1, bn1, relu, maxpool]   — frozen
  └── layer1                         — frozen
  └── layer2                         — frozen
  └── layer3                         — frozen
  └── layer4                         — fine-tuned (lr = 1e-4)
  └── fc: Linear(2048 → 42)          — trained from scratch (lr = 1e-3)
```

### 3.2 Training strategy

The backbone is initialised with ImageNet weights and most of it is frozen. Only the last residual block (`layer4`) and the new classification head (`fc`) are updated during training.

A differential learning rate scheme is applied via two parameter groups in the Adam optimiser:

- `layer4`: `lr = 1e-4` — fine-tuning of high-level features already learned from ImageNet.
- `fc`: `lr = 1e-3` — the head is initialised randomly and needs a faster learning rate.

**Loss function:** Cross-entropy loss.  
**Optimiser:** Adam.  
**Epochs:** 10.

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

*To be added.*

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
