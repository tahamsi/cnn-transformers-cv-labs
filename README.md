# cv-cnn-transformers-teaching

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

A master’s-level teaching repo for learning **CNNs** and **Vision Transformers** with hands-on labs. The materials emphasize intuition first, light math second, and clean PyTorch/Hugging Face implementations.

## Introduction
Convolutional Neural Networks (CNNs) dominated vision for a decade because they encode strong image priors: **locality** (nearby pixels matter most) and **translation equivariance** (a pattern can appear anywhere). Transformers started in NLP and then moved to vision by treating images as sequences of patches, enabling **global context** via self-attention. This repo teaches both paradigms, their tradeoffs, and how to use them in practice.

## CNNs: What they are and how they work
A CNN is a neural network built from **convolutions**, **nonlinearities**, and **pooling**:
- **Convolution** applies learnable filters (kernels) over local neighborhoods to detect patterns like edges or textures.
- **Weight sharing** means the same filter is used across the image, giving translation equivariance.
- **Pooling** (e.g., max-pool) reduces spatial resolution and builds robustness to small shifts.
- **Feature hierarchies** emerge: early layers detect edges, middle layers detect parts, deeper layers detect objects.

Mathematically, a 2D convolution computes a weighted sum over a local patch. Stacking layers increases the **receptive field**, allowing the model to integrate larger context. CNNs are data-efficient because their inductive bias aligns with natural images.

Classic CNN papers:
- LeNet (LeCun et al., 1998)
- AlexNet (Krizhevsky et al., 2012)
- VGG (Simonyan & Zisserman, 2014)
- ResNet (He et al., 2016)

## Vision Transformers: What they are and how they work
A Vision Transformer (ViT) treats an image as a sequence of **patch tokens**:
1. **Patchify** the image into fixed-size patches (e.g., 16x16).
2. **Embed** each patch with a linear projection.
3. Add **positional embeddings** to preserve spatial order.
4. Run **self-attention** blocks that let each patch attend to all others.
5. Use a **class token** (or pooled tokens) for classification.

Self-attention computes pairwise interactions between all tokens, allowing global context from the start. ViTs typically need large datasets or pretraining, but transfer learning with Hugging Face makes them practical on smaller datasets.

Key Transformer papers:
- Attention Is All You Need (Vaswani et al., 2017)
- ViT (Dosovitskiy et al., 2021)
- DeiT (Touvron et al., 2021)
- Swin Transformer (Liu et al., 2021)
- DETR (Carion et al., 2020)

## CNNs vs Transformers: Key differences
- **Inductive bias**: CNNs bake in locality and translation equivariance; Transformers are more flexible but data-hungry.
- **Context**: CNNs build global context gradually; Transformers use global attention from the start.
- **Compute**: CNNs are efficient for small/medium images; attention can be expensive for large resolutions.
- **Data needs**: CNNs work well with smaller datasets; ViTs often need large-scale pretraining.
- **Interpretability**: CNN filters are spatially local; attention maps provide global but diffuse explanations.

### Pros and cons (quick scan)
**CNN Pros**
- Efficient and data-friendly
- Strong spatial inductive bias
- Many mature architectures and pretrained weights

**CNN Cons**
- Limited global context early in the network
- Struggles with long-range dependencies unless deep

**ViT Pros**
- Global context via self-attention
- Scales well with large data and compute
- Strong transfer learning with pretrained checkpoints

**ViT Cons**
- Data-hungry without pretraining
- Attention is compute-heavy for high resolution

## Extensions and trends
- **DeiT** improves data efficiency for ViTs.
- **Swin** uses windowed attention for better scaling.
- **DETR** applies transformers to detection.
- Hybrid models mix CNN backbones with attention blocks.

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
- LeCun et al., 1998 — Gradient-Based Learning Applied to Document Recognition (LeNet)
- Krizhevsky et al., 2012 — ImageNet Classification with Deep CNNs (AlexNet)
- He et al., 2016 — Deep Residual Learning for Image Recognition (ResNet)
- Vaswani et al., 2017 — Attention Is All You Need
- Dosovitskiy et al., 2021 — An Image is Worth 16x16 Words (ViT)
- Touvron et al., 2021 — DeiT
- Liu et al., 2021 — Swin Transformer
- Carion et al., 2020 — DETR
- Hugging Face docs: https://huggingface.co/docs/transformers
- PyTorch vision docs: https://pytorch.org/vision/stable/

## License
MIT (see `LICENSE`).
