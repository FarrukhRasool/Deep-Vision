# Deep Vision: CNN vs Vision Transformer for Image Classification

A comparative study of **ResNet-50** and **Vision Transformer (ViT-B16)** for image classification using **Transfer Learning** on a small subset of the Food-101 dataset.

This project was completed as part of the **Deep Vision** course at **OTH Amberg-Weiden**.

---

## Project Overview

The objective of this project is to compare the performance of a traditional **Convolutional Neural Network (ResNet-50)** with a **Vision Transformer (ViT-B16)** when fine-tuned on a relatively small image dataset.

Instead of training models from scratch, ImageNet pretrained weights are used to investigate:

- Transfer learning performance
- Different classification head architectures
- Effect of regularization techniques
- Fine-tuning strategies
- CNN vs Vision Transformer comparison

---

## Dataset

The project uses a **10-class subset** of the **Food-101** dataset.

### Selected Classes

- Apple Pie
- Cup Cakes
- French Fries
- Hamburger
- Ice Cream
- Macarons
- Pizza
- Ramen
- Sushi
- Waffles

### Dataset Split

| Split | Images |
|--------|--------:|
| Training | 7,500 |
| Validation | 1,000 |
| Test | 1,500 |

Each class contains:

- 750 training images
- 100 validation images
- 150 test images

---

# Models Evaluated

## ResNet-50 Experiments

Six different architectures were evaluated.

| Model | Description |
|--------|-------------|
| M1 | ResNet-50 + Dense |
| M2 | Deep Neural Network Head |
| M3 | DNN + Dropout |
| M4 | DNN + Batch Normalization + Dropout |
| M5 | DNN + BatchNorm + Dropout + L2 Regularization |
| M6 | Fine-tuned ResNet-50 |

---

## Vision Transformer Experiments

Five ViT head configurations were evaluated.

| Model | Description |
|--------|-------------|
| V1 | Dense Classification Head |
| V2 | Deep Neural Network Head |
| V3 | DNN + Dropout |
| V4 | DNN + Dropout + L2 |
| V4.2 | Hyperparameter Tuned Final Model |

---

# Training Pipeline

### Image Size

```
224 × 224
```

### Batch Size

```
64
```

### Optimizer

```
Adam
```

### Loss Function

```
Sparse Categorical Crossentropy
```

### Data Augmentation

- Random Crop
- Random Horizontal Flip
- ImageNet preprocessing
- Shuffle
- Batch
- Prefetch

---

# Results

## ResNet-50

| Model | Validation Accuracy |
|--------|--------------------:|
| M1 | 91.12% |
| M2 | 91.22% |
| M3 | 92.24% |
| M4 | 91.94% |
| M5 | 92.04% |
| M6 (Fine-tuned) | **92.94%** |

Final Test Accuracy

**93%**

---

## Vision Transformer (ViT-B16)

| Model | Validation Accuracy |
|--------|--------------------:|
| V1 | 94.79% |
| V2 | 95.30% |
| V3 | **95.51%** |
| V4 | 94.89% |
| V4.2 | 93.97% |

Final Test Accuracy

**95%**

---

# Comparison

| Model | Test Accuracy |
|--------|--------------:|
| ResNet-50 | **93%** |
| ViT-B16 | **95%** |

The experiments demonstrate that **Vision Transformers outperform CNNs when using ImageNet pretrained weights**, even on relatively small datasets.

---

# Technologies Used

- Python
- TensorFlow
- TensorFlow Datasets (TFDS)
- Keras
- Keras Hub
- NumPy
- Matplotlib

---

# Repository Structure

```
.
├── DeepVision.ipynb          # Jupyter Notebook
├── DeepVision.py             # Python implementation
├── DeepVision.pdf            # Project report
├── README.md
├── models/                   # (ignored)
└── images/                   # Training plots (optional)
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/DeepVision.git
cd DeepVision
```

Install dependencies

```bash
pip install tensorflow tensorflow-datasets keras keras-hub matplotlib
```

Run

```bash
python DeepVision.py
```

or

```bash
jupyter notebook DeepVision.ipynb
```

---

# Large Model Files

The trained `.keras` models are **not included** in this repository because they exceed GitHub's file size limit.

To include trained models, use **Git Large File Storage (Git LFS)** or provide them through cloud storage.

---

# Key Findings

- Transfer learning significantly improves performance on small datasets.
- Deep classification heads improve task-specific learning.
- Dropout, Batch Normalization, and L2 Regularization reduce overfitting.
- Fine-tuning further improves CNN performance.
- Vision Transformers achieve better generalization than ResNet-50 under transfer learning.

---

# Authors

**Farrukh Rasool**


---

# License

This project was developed for academic and educational purposes.
