from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    name: str
    csv_path: str        # path to the fold/split CSV
    image_col: str       # column with the image file path
    label_col: str       # column with the class label
    split_col: str       # column that assigns each row to train / val / test
    train_splits: List[str]   # values in split_col that form the training set
    val_splits: List[str]     # values in split_col that form the validation set
    num_classes: int
    classes_csv: Optional[str] = None    # if set, load label→id mapping from here
    source_col: Optional[str] = None     # if set, use source_roots to resolve paths
    source_roots: Dict[str, str] = field(default_factory=dict)  # source → path prefix


# ---------------------------------------------------------------------------
# VippStar — original single-source dataset (~8 668 images, 42 classes)
# ---------------------------------------------------------------------------
VIPPSTAR = DataConfig(
    name="vippstar",
    csv_path="data/annotations/food_classification_dataset.csv",
    image_col="images",
    label_col="food_label",
    split_col="split",
    train_splits=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"],
    val_splits=["holdout_val", "holdout_test"],
    num_classes=42,
)


# ---------------------------------------------------------------------------
# Unified — three sources (vippstar + MedGR + odin), 249 classes, 3-fold CV
# Use one of the three folds below; the others rotate the test set.
# ---------------------------------------------------------------------------
def _unified(fold: int) -> DataConfig:
    return DataConfig(
        name=f"unified_fold_{fold}",
        csv_path=f"cnn_splits/fold_{fold}.csv",
        image_col="image_path",
        label_col="label",
        split_col="split",
        train_splits=["train"],
        val_splits=["val"],
        num_classes=249,
        classes_csv="cnn_splits/classes.csv",
        source_col="source",
        source_roots={
            "vippstar": "",            # paths already start with data/raw/
            "medgr": "data/external",
            "odin": "data/external",
        },
    )


UNIFIED_FOLD_0 = _unified(0)
UNIFIED_FOLD_1 = _unified(1)
UNIFIED_FOLD_2 = _unified(2)
