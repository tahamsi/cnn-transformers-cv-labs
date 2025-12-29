# cv-cnn-transformers-teaching

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

A master’s-level teaching repo for learning **CNNs** and **Vision Transformers** with hands-on labs. The materials emphasize intuition first, light math second, and clean PyTorch/Hugging Face implementations.

## Learning outcomes
- Explain CNN inductive biases (locality, translation equivariance) and receptive fields.
- Build and train a CNN from scratch on CIFAR-10/Tiny ImageNet.
- Explain ViT basics: patch embedding, positional encodings, self-attention, class tokens.
- Fine-tune pretrained ViT/Swin models with Hugging Face Trainer.
- Compare CNNs vs Transformers in terms of data efficiency, compute, and tradeoffs.
- Run small demos for classification, detection, and segmentation.

## Repository structure
```
cv-cnn-transformers-teaching/
  notebooks/
    00_Building_Blocks_From_Scratch.ipynb
    01_CNN_Fundamentals.ipynb
    02_Vision_Transformers_Fundamentals.ipynb
    03_HuggingFace_Vision_FineTuning.ipynb
    04_Comparison_CNN_vs_Transformers.ipynb
  scripts/
    download_data.py
    train_cnn.py
    train_vit_hf.py
    eval.py
    visualize.py
  src/
    data/
    models/
    trainers/
    utils/
    vision_utils/
  data/              # datasets (placeholder)
  outputs/           # plots/checkpoints/metrics (placeholder)
  requirements.txt
  LICENSE
```

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Colab notes
- Use **Runtime → Change runtime type → GPU**.
- Run the install cell at the top of each notebook.
- For speed, default training runs are short (1–3 epochs). See **Scale Up** sections to train longer.

## Data
Download and prepare datasets (Tiny ImageNet with CIFAR-10 fallback, Penn-Fudan, Oxford-IIIT Pet):
```bash
python scripts/download_data.py
```

## Scripts
- Train a CNN baseline:
```bash
python scripts/train_cnn.py --dataset tiny-imagenet --model simple --epochs 3
```
- Fine-tune a ViT with Hugging Face Trainer:
```bash
python scripts/train_vit_hf.py --dataset cifar10 --epochs 1
```
- Evaluate a checkpoint:
```bash
python scripts/eval.py --dataset cifar10 --model simple --checkpoint outputs/checkpoints/simple_cifar10.pt
```
- Visualize filters or attention:
```bash
python scripts/visualize.py --mode cnn
python scripts/visualize.py --mode vit
```

## Notebooks
- `notebooks/00_Building_Blocks_From_Scratch.ipynb` — step-by-step CNN and toy ViT from scratch (no Hugging Face).
- `notebooks/01_CNN_Fundamentals.ipynb` — CNN theory, training, filters, feature maps, and Grad-CAM.
- `notebooks/02_Vision_Transformers_Fundamentals.ipynb` — ViT concepts + Hugging Face fine-tuning.
- `notebooks/03_HuggingFace_Vision_FineTuning.ipynb` — end-to-end HF workflows, detection + segmentation demos.
- `notebooks/04_Comparison_CNN_vs_Transformers.ipynb` — conceptual + empirical comparison with discussion prompts.

## References (papers + docs)
- CNN classics: LeNet, AlexNet, VGG, ResNet
- Transformers: Attention Is All You Need
- Vision Transformer (ViT)
- DeiT
- Swin Transformer
- DETR (object detection transformer)
- Hugging Face docs: transformers, Trainer, vision tasks
- PyTorch vision docs

## License
MIT (see `LICENSE`).
