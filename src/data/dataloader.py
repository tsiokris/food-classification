from torch.utils.data import DataLoader, Dataset

def get_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

