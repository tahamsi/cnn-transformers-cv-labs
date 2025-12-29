"""Evaluate a classification model checkpoint and save metrics."""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch

from src.data.datasets import get_classification_dataloaders
from src.models.cnn import SimpleCNN, get_resnet18
from src.utils.device import get_device
from src.utils.metrics import confusion_matrix
from src.utils.plotting import plot_confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tiny-imagenet", choices=["tiny-imagenet", "cifar10"])
    parser.add_argument("--model", default="simple", choices=["simple", "resnet18"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    data_root = os.path.join(ROOT, "data")
    _, val_loader, num_classes, class_names = get_classification_dataloaders(
        data_root,
        args.dataset,
        batch_size=64,
        image_size=args.image_size,
    )

    if args.model == "simple":
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = get_resnet18(num_classes=num_classes, pretrained=False)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    cm = confusion_matrix(y_true, y_pred, num_classes)
    acc = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)

    outputs_dir = os.path.join(ROOT, "outputs")
    os.makedirs(os.path.join(outputs_dir, "metrics"), exist_ok=True)
    metrics_path = os.path.join(outputs_dir, "metrics", f"metrics_{args.model}_{args.dataset}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc}, f, indent=2)

    plot_confusion_matrix(
        cm,
        class_names,
        title="Confusion Matrix",
        save_path=os.path.join(outputs_dir, "plots", f"cm_{args.model}_{args.dataset}.png"),
    )


if __name__ == "__main__":
    main()
