"""Visualization helper script for filters, feature maps, and attention."""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from src.models.cnn import SimpleCNN
from src.models.vit_toy import ToyViT
from src.vision_utils.attention import attention_rollout


def visualize_cnn_filters(model: SimpleCNN, save_path: str) -> None:
    weights = model.features[0].weight.data.clone()
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    fig, axes = plt.subplots(4, 8, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        if i < weights.size(0):
            img = weights[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_vit_attention(model: ToyViT, image: torch.Tensor, save_path: str) -> None:
    model.eval()
    with torch.no_grad():
        _, attn_maps = model(image.unsqueeze(0), return_attn=True)
    rollout = attention_rollout(attn_maps)[0]
    if model.use_cls_token:
        rollout = rollout[0, 1:]
    side = int(np.sqrt(rollout.numel()))
    attn_map = rollout.reshape(side, side).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(attn_map, cmap="inferno")
    plt.title("Attention Rollout")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cnn", "vit"], default="cnn")
    args = parser.parse_args()

    outputs_dir = os.path.join(ROOT, "outputs", "plots")
    os.makedirs(outputs_dir, exist_ok=True)

    if args.mode == "cnn":
        model = SimpleCNN(num_classes=10)
        visualize_cnn_filters(model, os.path.join(outputs_dir, "cnn_filters.png"))
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
        image, _ = dataset[0]
        model = ToyViT(img_size=32, patch_size=4, embed_dim=128, depth=2, num_heads=4, num_classes=10)
        visualize_vit_attention(model, image, os.path.join(outputs_dir, "vit_attention.png"))


if __name__ == "__main__":
    main()
