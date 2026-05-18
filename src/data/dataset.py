import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodDataset(Dataset):

    def __init__(self, food_dataset_csv, transform=None, split=None):
        self.df = pd.read_csv(food_dataset_csv)
        all_labels = sorted(self.df["food_label"].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        if split:
            if isinstance(split, str):
                split = [split]
            self.df = self.df[self.df["split"].isin(split)]
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["images"]
        label = self.df.iloc[idx]["food_label"]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)

        return img, self.label_to_idx[label]