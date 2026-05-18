import torchvision.models as models
from torch import nn


class FoodClassifier(nn.Module):

    def __init__(self, num_classes: int, backbone: str, freeze_backbone: bool):

        super().__init__()
        self.model = getattr(models, backbone)(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)