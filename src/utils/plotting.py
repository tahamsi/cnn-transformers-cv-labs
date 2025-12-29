"""Matplotlib helpers for common plots."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_curves(history: Dict[str, List[float]], title: str, save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 4))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm: torch.Tensor, class_names: List[str], title: str, save_path: Optional[str] = None) -> None:
    cm_np = cm.cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_np, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_image_grid(images: torch.Tensor, title: str, save_path: Optional[str] = None) -> None:
    grid = torch.clamp(images, 0, 1)
    n = int(np.ceil(np.sqrt(grid.size(0))))
    fig, axes = plt.subplots(n, n, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < grid.size(0):
            img = grid[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
