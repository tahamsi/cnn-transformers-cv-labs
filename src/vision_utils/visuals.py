"""Visualization helpers for bounding boxes and masks."""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_bboxes(image: np.ndarray, boxes: List[List[float]], labels: List[str], title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, label, color="white", fontsize=8, bbox=dict(facecolor="black", alpha=0.6))
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def overlay_mask(image: np.ndarray, mask: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image)
    ax.imshow(mask, alpha=0.5, cmap="jet")
    ax.set_title(title)
    ax.axis("off")
    plt.show()
