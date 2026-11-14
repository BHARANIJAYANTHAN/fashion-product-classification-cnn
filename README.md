# Automated Fashion Product Classification for E-commerce using CNN

**Deep Learning – Application-Oriented Project**
SRM Institute of Science and Technology · Department of CINTEL · Branch AI-D

| | |
|---|---|
| Student | Bharani Jayanthan K S |
| Register No. | RA2211047010109 |
| Faculty | Dr. Vinoth |

## Problem statement
E-commerce platforms receive thousands of new product photos every day, and categorising them by hand is slow and error-prone. This project builds a **Convolutional Neural Network (CNN)** that automatically classifies a product image into one of 10 fashion categories, so catalogues can be tagged instantly.

## Dataset
**Fashion-MNIST** (Zalando Research), loaded directly with `keras.datasets.fashion_mnist`, so no manual download is needed.
- 70,000 grey-scale images, 28 × 28 pixels (60,000 train / 10,000 test)
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Source: https://github.com/zalandoresearch/fashion-mnist

## Model architecture
```
Input 28×28×1
→ Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
→ Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
→ Flatten → Dense(128, ReLU) → Dropout(0.4) → Dense(10, Softmax)
```
Optimiser: Adam · Loss: sparse categorical cross-entropy · Early Stopping (patience 3) · up to 10 epochs, batch size 128

## Syllabus units applied
| Unit | Concepts |
|---|---|
| I | Supervised learning, train/validation/test split, over-fitting, regularization |
| II | Dense layers, ReLU/Softmax, cross-entropy, back-propagation, Adam, Batch Normalization, Dropout, Early Stopping |
| III | Convolution, pooling, CNNs for computer vision |
| V | Deep architectures for computer vision |

## How to run
1. Open `Fashion_Product_Classification_CNN.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook, or open it from this repo).
2. Runtime → Change runtime type → **T4 GPU** (optional; CPU also works).
3. Runtime → **Run all**. It takes about 2 minutes on a GPU.

Requirements (pre-installed in Colab): Python 3, TensorFlow 2.x, NumPy, Matplotlib, Seaborn, scikit-learn.

## Results
Output screenshots and final accuracy are in the `screenshots/` folder and the executed notebook.

## Repository structure
```
├── Fashion_Product_Classification_CNN.ipynb   # complete source code with outputs
├── screenshots/                               # output screenshots
└── README.md
```
