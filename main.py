import torch
from torch import nn
from src.data.dataset import FoodDataset
from src.data.dataloader import get_dataloader
from src.data.transforms import train_transforms, val_transforms
from src.models.model import FoodClassifier
from src.training.train import train, validate


# ----------------------- CONFIG --------------------------
PATH = "data/annotations/food_classification_dataset.csv"
LOSS_FN = nn.CrossEntropyLoss()
NUM_CLASSES = 42
BACKBONE = "resnet50"
FREEZE_BACKBONE = True
EPOCHS = 10


# ----------------------- MAIN --------------------------
train_dataset = FoodDataset(food_dataset_csv=PATH, transform=train_transforms, split=['fold_0','fold_1','fold_2','fold_3','fold_4'])
val_dataset = FoodDataset(food_dataset_csv=PATH, transform=val_transforms, split=['holdout_val','holdout_test'])

train_dataloader = get_dataloader(dataset=train_dataset)
val_dataloader = get_dataloader(dataset=val_dataset, shuffle=False)

model = FoodClassifier(num_classes=NUM_CLASSES, backbone=BACKBONE, freeze_backbone=FREEZE_BACKBONE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model=model, dataloader=train_dataloader, epochs=EPOCHS, loss_fn=LOSS_FN, optimizer=optimizer)
validate(model=model, dataloader=val_dataloader, loss_fn=LOSS_FN)

torch.save(model.state_dict(), "models/model.pth")