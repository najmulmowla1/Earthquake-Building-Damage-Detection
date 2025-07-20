# Earthquake-Building-Damage-Detection

This repository provides the core components of a deep learning framework designed for automatic post-earthquake building damage detection using high-resolution UAV imagery.

The Proposed STCHMDA-CVT architecture, integrates a custom Convolutional Neural Network (CNN) backbone with a Vision Transformer (ViT), enhanced by a novel Sparse Cross-Attention Hybrid Multi-Dimensional Attention (SCA HMDA) module. This hybrid structure captures both local and global dependencies in spatial feature maps for robust classification.

This work was motivated by the 2023 Turkey earthquakes, with the goal of supporting rapid and reliable structural damage assessment from aerial UAV data.

---

## 🔍 Dataset: UAVs-TEBDE

We introduce the **UAVs-Turkey Earthquake Building Damage Estimation (UAVs-TEBDE)** dataset:
- 3 damage categories: **Cracked**, **Partially Damaged**, and **Collapsed**
- 2,160 high-resolution samples per class (augmented)
- Image size: 256×256 pixels (RGB)
- Designed to support machine learning models in disaster assessment
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## 🧠 Model Components

This repository currently includes the following core modules:

### 1. `attention.py` – SCA_HMDA Module
- Multi-headed sparse cross-attention mechanism
- Combines spatial, temporal, and cross-modal representations
- Layer-normalized fusion of local and global features

### 2. `vit.py` – Vision Transformer Block
- Patch-based feature tokenization
- Positional encoding + stacked SCA_HMDA layers
- Reduces to compact global descriptors

### 3. 🧱 `model.py` – Full Architecture
- Custom CNN for initial feature extraction
- ViT block integrated into the pipeline
- Fully connected layers for classification

---

## 📦 Installation

Ensure you have the following dependencies installed:

```bash
pip install tensorflow keras albumentations
