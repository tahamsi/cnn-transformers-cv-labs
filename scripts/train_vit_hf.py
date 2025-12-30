"""Fine-tune a ViT model with Hugging Face Trainer."""

import argparse
import os
import sys

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Image, load_dataset
from torchvision import datasets as tv_datasets
from transformers import (AutoImageProcessor, Trainer, TrainingArguments,
                          ViTForImageClassification)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "tiny-imagenet"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token")
    parser.add_argument("--no-auth", action="store_true", help="Disable HF auth and use public access")
    args = parser.parse_args()

    data_root = os.path.join(ROOT, "data")

    if args.dataset == "tiny-imagenet":
        tiny_root = os.path.join(data_root, "tiny-imagenet-200")
        if os.path.isdir(tiny_root):
            dataset = load_dataset("imagefolder", data_dir=tiny_root)
        else:
            print("Tiny ImageNet not found; falling back to CIFAR-10")
            dataset = load_dataset("cifar10")
    else:
        dataset = load_dataset("cifar10")

    if "image" not in dataset["train"].features:
        print("HF dataset cache missing image column; falling back to torchvision CIFAR-10.")
        train_tv = tv_datasets.CIFAR10(root=data_root, train=True, download=True)
        test_tv = tv_datasets.CIFAR10(root=data_root, train=False, download=True)
        labels = list(train_tv.classes)
        features = Features(
            {
                "image": Image(),
                "label": ClassLabel(names=labels),
            }
        )

        train_dict = {"image": [img for img, _ in train_tv], "label": [lbl for _, lbl in train_tv]}
        test_dict = {"image": [img for img, _ in test_tv], "label": [lbl for _, lbl in test_tv]}
        dataset = {
            "train": Dataset.from_dict(train_dict, features=features),
            "test": Dataset.from_dict(test_dict, features=features),
        }

    if args.no_auth:
        token = False
    else:
        token = args.hf_token or os.getenv("HF_TOKEN") or None

    processor = AutoImageProcessor.from_pretrained(args.model, token=token)

    def preprocess(example):
        image = example["image"]
        inputs = processor(image, return_tensors="pt")
        if isinstance(image, list):
            return {
                "pixel_values": inputs["pixel_values"],
                "labels": example["label"],
            }
        return {
            "pixel_values": inputs["pixel_values"][0],
            "labels": example["label"],
        }

    if isinstance(dataset, dict):
        dataset = {k: v.with_transform(preprocess) for k, v in dataset.items()}
        label_names = dataset["train"].features["label"].names
    else:
        dataset = dataset.with_transform(preprocess)
        label_names = dataset["train"].features["label"].names

    model = ViTForImageClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)},
        token=token,
    )

    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    output_dir = os.path.join(ROOT, "outputs", "hf_vit")
    args_train = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test") or dataset.get("validation"),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
