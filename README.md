# Earthquake-Building-Damage-Detection

This repository provides the core components of a deep learning framework designed for automatic post-earthquake building damage detection using high-resolution UAV imagery.

The Proposed STCHMDA-CVT architecture integrates a custom Convolutional Neural Network (CNN) backbone with a Vision Transformer (ViT), enhanced by a novel Sparse Cross-Attention Hybrid Multi-Dimensional Attention (SCA HMDA) module. This hybrid structure captures both local and global dependencies in spatial feature maps for robust classification.

This work was motivated by the 2023 Turkey earthquakes, with the goal of supporting rapid and reliable structural damage assessment from aerial UAV data.

---

## Dataset: UAVs-TEBDE

We introduce the UAVs-Turkey Earthquake Building Damage Estimation (UAVs-TEBDE) dataset, a high-resolution aerial imagery collection developed to support AI-based post-disaster damage assessment using deep learning and computer vision techniques. The dataset comprises:

-Three damage categories: Cracked, Partially Damaged, and Collapsed
-2,160 augmented samples per class, totaling 6,480 images
-Image resolution: 256×256 pixels (RGB)
-Curated for training and benchmarking machine learning models in structural damage classification

Distributed under the CC BY 4.0 license

The UAVs-TEBDE dataset is publicly available via Mendeley Data:
https://data.mendeley.com/drafts/5m349hfvkb
Citation:
Mowla, Md. Najmul; Asadi, Davood (2025), UAVs-based Turkey Earthquake Building Damage Estimation Dataset (UAVs-TEBDE), Mendeley Data, V3, https://doi.org/10.17632/5m349hfvkb.3

## Model Components

This repository currently includes the following core modules:

### 1. `attention.py` – SCA_HMDA Module
- Multi-headed sparse cross-attention mechanism
- Combines spatial, temporal, and cross-modal representations
- Layer-normalized fusion of local and global features

### 2. `vit.py` – Vision Transformer Block
- Patch-based feature tokenization
- Positional encoding + stacked SCA_HMDA layers
- Reduces to compact global descriptors

### 3. `model.py` – Full Architecture
- Custom CNN for initial feature extraction
- ViT block integrated into the pipeline
- Fully connected layers for classification

---

## Installation

Ensure you have the following dependencies installed:

```bash
pip install tensorflow keras albumentations
