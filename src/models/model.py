import torchvision.models as models
from torch import nn


ARCH_CONFIGS = {
    "resnet50": {
        "in_features": 2048,
        "head": "fc",
        "last_block": "layer4",
    },
    "efficientnet_b3": {
        "in_features": 1536,
        "head": ("classifier", 1),
        "last_block": ("features", -1),
    },
    "mobilenet_v3_large": {
        "in_features": 1280,
        "head": ("classifier", 3),
        "last_block": ("features", -1),
    },
    "densenet121": {
        "in_features": 1024,
        "head": "classifier",
        "last_block": "features.denseblock4",
    },
}


def _get_submodule(model, spec):
    """Get a submodule by string attr (optionally dotted) or (attr, index) tuple."""
    if isinstance(spec, tuple):
        attr, idx = spec
        return getattr(model, attr)[idx]
    m = model
    for part in spec.split("."):
        m = getattr(m, part)
    return m


def _set_submodule(model, spec, new_module):
    """Set a submodule by string attr or (attr, index) tuple."""
    if isinstance(spec, tuple):
        attr, idx = spec
        getattr(model, attr)[idx] = new_module
    else:
        setattr(model, spec, new_module)


class FoodClassifier(nn.Module):

    def __init__(self, num_classes: int, backbone: str, freeze_backbone: bool, unfreeze_last_block: bool = False):
        super().__init__()
        cfg = ARCH_CONFIGS[backbone]
        self.model = getattr(models, backbone)(weights="IMAGENET1K_V1")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if unfreeze_last_block:
            last_block = _get_submodule(self.model, cfg["last_block"])
            for param in last_block.parameters():
                param.requires_grad = True

        _set_submodule(self.model, cfg["head"], nn.Linear(cfg["in_features"], num_classes))

    def forward(self, x):
        return self.model(x)
