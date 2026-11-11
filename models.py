"""
Model architecture for the water accumulation branch.

Not full segmentation -- per the 2-month scope decision, water coverage
is estimated with a lightweight patch classifier over the platform ROI
rather than a pixel-wise segmentation model. MobileNetV2 was chosen
because it was already the right size for a laptop/edge-device demo
before edge deployment was even on the table.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class WaterClassifier(nn.Module):
    """Binary classifier: water / no-water, applied to a cropped patch of
    the platform surface ROI (not the full frame)."""

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained_backbone else None
        self.backbone = mobilenet_v2(weights=weights)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_water_classifier(weights_path: str, device: str = "cpu") -> WaterClassifier:
    model = WaterClassifier(pretrained_backbone=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Standalone check: does the architecture build and run a forward pass
    # on a dummy tensor? This does NOT need real trained weights.
    model = WaterClassifier(pretrained_backbone=False)
    dummy_input = torch.randn(1, 3, 224, 224)  # MobileNetV2's expected input size
    output = model(dummy_input)
    print("WaterClassifier output shape:", output.shape)  # expect (1, 2)
    print("Architecture OK.")
