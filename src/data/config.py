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
def _rotation(fold: int) -> DataConfig:
    return DataConfig(
        name=f"rotation_fold{fold}",
        csv_path="data/annotations/master_splits.csv",
        image_col="image_path",
        label_col="label",
        split_col="fold_split",
        train_splits=[f"fold_{fold}_train"],
        val_splits=[f"fold_{fold}_val"],
        test_splits=[f"fold_{fold}_test"],
        num_classes=216,
        classes_csv="cnn_splits/classes.csv",
        source_col="source",
        source_roots={
            "vippstar": "",
            "medgr": "data/external",
            "odin": "data/external",
        },
    )


ROTATION_0 = _rotation(fold=0)
ROTATION_1 = _rotation(fold=1)
ROTATION_2 = _rotation(fold=2)
