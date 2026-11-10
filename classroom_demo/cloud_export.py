"""Export the frozen, project ResNet18 feature extractor for the small cloud instance."""
from pathlib import Path
import sys

import torch
from torchvision.models import resnet18


class Features(torch.nn.Module):
    def __init__(self, weights: Path):
        super().__init__()
        self.backbone = resnet18()
        self.backbone.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))

    def forward(self, image):
        net = self.backbone
        image = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        layer2 = net.layer2(net.layer1(image))
        return layer2, net.layer3(layer2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    net = Features(Path(sys.argv[1])).eval()
    torch.onnx.export(net, torch.zeros((1, 3, 128, 128)), sys.argv[2], opset_version=13,
                      dynamo=False, input_names=["image"], output_names=["layer2", "layer3"])
