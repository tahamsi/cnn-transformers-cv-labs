"""Minimal PyTorch training utilities."""

from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.metrics import accuracy_topk


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy_topk(logits, labels, (1,))["top1"]
        total_loss += loss.item()
        total_acc += acc

    return {
        "loss": total_loss / len(loader),
        "acc": total_acc / len(loader),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            acc = accuracy_topk(logits, labels, (1,))["top1"]
            total_loss += loss.item()
            total_acc += acc

    return {
        "loss": total_loss / len(loader),
        "acc": total_acc / len(loader),
    }
