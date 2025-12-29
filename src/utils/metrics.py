"""Basic metrics without external dependencies."""

from typing import Dict, List, Tuple

import torch


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Dict[str, float]:
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        results[f"top{k}"] = correct_k / batch_size
    return results


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(cm: torch.Tensor) -> Dict[str, Dict[str, float]]:
    report = {}
    num_classes = cm.size(0)
    for i in range(num_classes):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        report[str(i)] = {"precision": precision, "recall": recall, "f1": f1}
    return report
