"""Train a CNN/ResNet on Tiny ImageNet or CIFAR-10."""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
from torch import nn
from torch.optim import Adam

from src.data.datasets import get_classification_dataloaders
from src.models.cnn import SimpleCNN, get_resnet18
from src.trainers.training import evaluate, train_one_epoch
from src.utils.device import get_device
from src.utils.plotting import plot_curves
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tiny-imagenet", choices=["tiny-imagenet", "cifar10"])
    parser.add_argument("--model", default="simple", choices=["simple", "resnet18"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    data_root = os.path.join(ROOT, "data")
    train_loader, val_loader, num_classes, _ = get_classification_dataloaders(
        data_root,
        args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    if args.model == "simple":
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = get_resnet18(num_classes=num_classes, pretrained=True)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate(model, val_loader, device)
        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])
        print(f"Epoch {epoch+1}: train acc {train_stats['acc']:.3f} val acc {val_stats['acc']:.3f}")

    outputs_dir = os.path.join(ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    ckpt_path = os.path.join(outputs_dir, "checkpoints", f"{args.model}_{args.dataset}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    plot_curves(
        {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
        title="Loss Curves",
        save_path=os.path.join(outputs_dir, "plots", f"loss_{args.model}_{args.dataset}.png"),
    )


if __name__ == "__main__":
    main()
