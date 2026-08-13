# CelebA Facial Attribute Recognition

**Third Prize — XJTLU Facial Recognition Contest**

A PyTorch project for **multi-label facial attribute recognition** on the CelebA dataset. Given one aligned face image, the model predicts the 40 binary attributes provided by CelebA, such as *Smiling*, *Eyeglasses*, *Male*, *Young*, and hair-related attributes.

> Although the repository originated from a facial-recognition competition, the implemented learning task is specifically **40-attribute facial analysis**, not face-identity verification.

## Project Overview

The project explores how model capacity changes performance on the same facial-attribute prediction task. It contains three model families:

| Model | Role | Main idea |
| --- | --- | --- |
| `SimpleNN` | Baseline | Flatten the image and learn attribute logits with a two-layer MLP |
| `SimpleCNN` | Lightweight vision model | Learn local facial features with convolution blocks and adaptive pooling |
| `ResNet50` | Transfer-learning model | Fine-tune an ImageNet-pretrained ResNet-50 for 40 binary outputs |

The competition project received **Third Prize in the XJTLU Facial Recognition Contest**.

## Problem Formulation

Each CelebA image is associated with 40 binary attributes. Therefore, this is not a 40-class softmax problem: one face can have many positive attributes simultaneously.

The network outputs a vector of 40 logits:

```text
face image
    ↓
feature extractor
    ↓
40 independent attribute logits
    ↓ sigmoid at evaluation
40 binary predictions
```

Training uses `BCEWithLogitsLoss`, which combines the numerically stable sigmoid operation with binary cross-entropy for multi-label learning.

## Modeling Strategy

### 1. Fully Connected Baseline — `SimpleNN`

The simplest baseline flattens the normalized image and feeds it through two fully connected layers. It serves as a reference point for understanding why spatial inductive bias matters in face analysis.

### 2. Lightweight CNN — `SimpleCNN`

The CNN progressively extracts local facial patterns through three convolution blocks:

```text
Conv → BatchNorm → ReLU → MaxPool
Conv → BatchNorm → ReLU → MaxPool
Conv → BatchNorm → ReLU → MaxPool
               ↓
       AdaptiveAvgPool
               ↓
          Linear(40)
```

Adaptive pooling removes the original dependence on a hard-coded flattened feature size and makes the architecture more robust to image-resolution changes.

### 3. ResNet50 Transfer Learning

For the strongest model family in the repository, a pretrained ResNet-50 is adapted by replacing its final classification layer with a 40-output linear head. This allows the model to reuse general visual representations while learning CelebA-specific facial attributes.

## Data Pipeline

The repository uses CelebA's official:

- aligned face images;
- train/validation/test partition file;
- 40-attribute annotation file.

The loader converts CelebA labels from `{-1, +1}` to `{0, 1}` and applies ImageNet-style normalization. Training additionally uses random horizontal flipping as lightweight augmentation.

A fixed resize to `156 × 128` keeps the data shape deterministic across environments.

## Training and Evaluation

The training pipeline now includes:

- deterministic random seeds;
- GPU/CPU device selection;
- `BCEWithLogitsLoss` for stable multi-label optimization;
- AdamW or SGD optimization;
- validation-based best-checkpoint selection;
- label-level accuracy;
- micro precision, recall, and F1;
- held-out test evaluation;
- saved checkpoints, metrics, training history, and learning curves.

The public repository does **not** currently contain a verified competition-score artifact, so no unverified accuracy number is reported here. The externally confirmed result is the **Third Prize** competition award.

## Repository Structure

```text
face-recognition/
├── Dataloader.py     # CelebA parsing, splits, transforms, multi-label targets
├── Model.py          # SimpleNN, SimpleCNN, ResNet50 model factory
├── Train.py          # training, validation, test evaluation, checkpointing
├── requirements.txt  # Python dependencies
├── .gitignore        # generated files / local data exclusions
└── README.md
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Point the project to CelebA either with `--data-root` or the environment variable:

```bash
export CELEBA_ROOT=/path/to/CelebA
```

Expected layout:

```text
CelebA/
└── Dataset/
    ├── Eval/list_eval_partition.txt
    ├── Anno/list_attr_celeba.txt
    └── Img/img_align_celeba/
```

Train a model, for example:

```bash
python Train.py --model ResNet50 --epochs 25 --batch-size 128
```

Outputs are written to `outputs/` by default.

## Engineering Improvements in This Version

The original competition code demonstrated the core modeling pipeline, but several details were tied to the original development machine. This repository refresh keeps the same task and model progression while making the code easier to inspect and reproduce:

- removed the hard-coded Windows data path;
- added `CELEBA_ROOT` / `--data-root` configuration;
- made CelebA annotation parsing explicit and robust;
- enforced deterministic image dimensions;
- changed `Sigmoid + BCELoss` to `BCEWithLogitsLoss`;
- removed hard-coded CNN flatten dimensions with adaptive pooling;
- replaced the legacy `torch.hub` ResNet loader with `torchvision.models`;
- added reproducible evaluation and checkpoint artifacts.

## Award

**Third Prize, XJTLU Facial Recognition Contest**

The project demonstrates a complete computer-vision workflow from dataset parsing and multi-label formulation to baseline comparison, CNN feature learning, transfer learning, and held-out evaluation.

## Dataset

CelebA: *Large-scale CelebFaces Attributes (CelebA) Dataset*, originally released by the Multimedia Laboratory at The Chinese University of Hong Kong.

Dataset page: <https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
