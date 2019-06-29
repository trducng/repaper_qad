"""Retrieve appropriate feature extractor"""
import torch.nn as nn
from torchvision import models


def get_densenet_3blocks():
    """Get DenseNet
    
    # Returns
        [nn.Module]: the feature extractor
        [int]: the number of output channels
        [int]: the height of output channels
    """

    model = models.DenseNet(
            growth_rate=32, block_config=(6, 12, 32),
            num_init_features=64, drop_rate=0.1)
    features = nn.Sequential(
        model.features,
        nn.ReLU()
    )

    return features, 1280, 4


def get_resnet101():
    """Get ResNet 101 feature extractor

    # Returns
        [nn.Module]: the feature extractor
        [int]: the number of output channels
        [int]: the height of output channels
    """
    model = models.resnet101()
    features = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,

        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )

    return features, 2048, 4

