"""Model definitions for CNNs and toy ViT."""

from .cnn import SimpleCNN, get_resnet18
from .vit_toy import ToyViT

__all__ = ["SimpleCNN", "get_resnet18", "ToyViT"]
