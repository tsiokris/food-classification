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
        # features[6] and [7] are the last two real MBConv stages; [8] is the final projection conv
        "last_block": [("features", 6), ("features", 7), ("features", 8)],
    },
    "mobilenet_v3_large": {
        "in_features": 1280,
        "head": ("classifier", 3),
        # features[13]-[15] are the last InvertedResidual blocks; [16] is the final projection conv
        "last_block": [("features", 13), ("features", 14), ("features", 15), ("features", 16)],
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
            specs = cfg["last_block"]
            if not isinstance(specs, list):
                specs = [specs]
            for spec in specs:
                for param in _get_submodule(self.model, spec).parameters():
                    param.requires_grad = True

        _set_submodule(self.model, cfg["head"], nn.Linear(cfg["in_features"], num_classes))

    def forward(self, x):
        return self.model(x)
