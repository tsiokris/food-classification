import torchvision.models as models
from torch import nn


class FoodClassifier(nn.Module):

    def __init__(self, num_classes: int, backbone: str, freeze_backbone: bool, unfreeze_last_block: bool = False):

        super().__init__()
        self.model = getattr(models, backbone)(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        if unfreeze_last_block:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)