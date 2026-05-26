import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from src.data.config import DataConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path: str) -> Image.Image:
    """Load a .jpg or .npy file and return an RGB PIL Image."""
    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


class FoodDataset(Dataset):

    def __init__(self, config: DataConfig, split: str, transform=None):
        """
        Args:
            config:    DataConfig preset (e.g. VIPPSTAR, UNIFIED_FOLD_0).
            split:     "train" or "val" — resolved to CSV values via the config.
            transform: torchvision transform pipeline.
        """
        self.config = config
        self.transform = transform

        df = pd.read_csv(config.csv_path)

        allowed = config.train_splits if split == "train" else config.val_splits
        self.df = df[df[config.split_col].isin(allowed)].reset_index(drop=True)

        if config.classes_csv:
            classes_df = pd.read_csv(config.classes_csv)
            self.label_to_idx = dict(zip(classes_df["name"], classes_df["class_id"]))
        else:
            all_labels = sorted(df[config.label_col].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(self._resolve_path(row))
        if self.transform:
            img = self.transform(img)
        return img, self.label_to_idx[row[self.config.label_col]]

    def _resolve_path(self, row) -> str:
        path = row[self.config.image_col]
        if self.config.source_col:
            root = self.config.source_roots.get(row[self.config.source_col], "")
            return os.path.join(root, path) if root else path
        return path
