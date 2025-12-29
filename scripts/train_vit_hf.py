"""Fine-tune a ViT model with Hugging Face Trainer."""

import argparse
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
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

    processor = AutoImageProcessor.from_pretrained(args.model)

    def preprocess(example):
        image = example["image"]
        inputs = processor(image, return_tensors="pt")
        example["pixel_values"] = inputs["pixel_values"][0]
        example["labels"] = example["label"]
        return example

    dataset = dataset.with_transform(preprocess)
    label_names = dataset["train"].features["label"].names

    model = ViTForImageClassification.from_pretrained(
        args.model,
        num_labels=len(label_names),
        id2label={i: l for i, l in enumerate(label_names)},
        label2id={l: i for i, l in enumerate(label_names)},
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
