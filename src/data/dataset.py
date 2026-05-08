import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class FoodDataset(Dataset):

    def __init__(self, food_dataset_csv, transform=None):
        self.df = pd.read_csv(food_dataset_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["images"]
        label = self.df.iloc[idx]["food_label"]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)

        return img, label