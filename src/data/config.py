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
    test_splits: List[str] = field(default_factory=list)  # values in split_col that form the test set
    classes_csv: Optional[str] = None    # if set, load label→id mapping from here
    source_col: Optional[str] = None     # if set, use source_roots to resolve paths
    source_roots: Dict[str, str] = field(default_factory=dict)  # source → path prefix


# ---------------------------------------------------------------------------
# 3-fold CV with full rotation — each fold plays train / val / test in turn.
# Source CSV: data/annotations/master_splits.csv (run scripts/build_master_splits.py).
# holdout_final rows are present in the CSV but excluded from all three splits;
# they are reserved for a single final evaluation after model selection.
# ---------------------------------------------------------------------------
def _rotation(train_fold: int, val_fold: int, test_fold: int) -> DataConfig:
    return DataConfig(
        name=f"rotation_train{train_fold}_val{val_fold}_test{test_fold}",
        csv_path="data/annotations/master_splits.csv",
        image_col="image_path",
        label_col="label",
        split_col="fold",
        train_splits=[f"fold_{train_fold}"],
        val_splits=[f"fold_{val_fold}"],
        test_splits=[f"fold_{test_fold}"],
        num_classes=249,
        classes_csv="cnn_splits/classes.csv",
        source_col="source",
        source_roots={
            "vippstar": "",
            "medgr": "data/external",
            "odin": "data/external",
        },
    )


ROTATION_0 = _rotation(train_fold=0, val_fold=1, test_fold=2)
ROTATION_1 = _rotation(train_fold=1, val_fold=2, test_fold=0)
ROTATION_2 = _rotation(train_fold=2, val_fold=0, test_fold=1)
