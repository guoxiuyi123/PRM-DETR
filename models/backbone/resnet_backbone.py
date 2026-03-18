import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的 ResNet18 作为轻量级 Backbone (不下载权重，随机初始化)
        resnet = resnet18(weights=None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        features = []
        x = self.stem(x)
        x = self.layer1(x)
        features.append(x)  # P2: [B, 64, H/4, W/4]
        x = self.layer2(x)
        features.append(x)  # P3: [B, 128, H/8, W/8]
        x = self.layer3(x)
        features.append(x)  # P4: [B, 256, H/16, W/16]
        x = self.layer4(x)
        features.append(x)  # P5: [B, 512, H/32, W/32]
        return features
