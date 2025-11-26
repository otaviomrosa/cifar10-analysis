# CIFAR-10 CNN: Robustness, Uncertainty, and OOD Analysis

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. Beyond standard training, this notebook conducts a series of experiments to analyze the model's stochastic behavior, confidence calibration (entropy), and robustness against Out-of-Distribution (OOD) inputs like Gaussian noise.

## Project Overview

The objective of this project is not just to train a classifier, but to investigate the internal behaviors of deep learning models regarding:
1.  **Stochasticity:** How weight initialization and data shuffling affect convergence.
2.  **Confidence vs. Accuracy:** Correlating prediction entropy with model correctness.
3.  **Out-of-Distribution Behavior:** How the model handles unseen classes (CIFAR-100) and pure noise.

## Tech Stack

* **Language:** Python 3.x
* **Framework:** PyTorch, Torchvision
* **Analysis:** NumPy, Matplotlib

## Model Architecture

A standard lightweight CNN architecture was used for all experiments:
* **Conv Layer 1:** 3 input channels $\rightarrow$ 6 output channels, 5x5 kernel.
* **Pool:** Max Pooling (2x2).
* **Conv Layer 2:** 6 input channels $\rightarrow$ 16 output channels, 5x5 kernel.
* **Fully Connected Layers:** 120 $\rightarrow$ 84 $\rightarrow$ 10 outputs.
* **Activation:** ReLU.

## Experiments & Results

### Task 1: The Effects of Randomness
Two identical network architectures (`net1` and `net2`) were initialized with different random weights and trained on the same dataset with different mini-batch shuffling orders.

* **Training:** 5 Epochs, SGD (lr=0.001, momentum=0.9).
* **Result:**
    * Top-1 Accuracy: ~59% (both models).
    * **Agreement Score:** ~57.6%.
* **Conclusion:** ~42% of prediction errors are uncorrelated, driven purely by stochastic initialization and data ordering.

### Task 2: Confidence Analysis (Entropy)
We analyzed the model's "confidence" using the Shannon Entropy of the Softmax output distribution:

$$H(P) = -\sum_{i=1}^{K} p_i \log(p_i)$$

* **Findings:** An inverse relationship exists between accuracy and entropy.
* **High Confidence/High Accuracy:** 'Car' class (Acc: 94%, Entropy: 0.48).
* **Low Confidence/Low Accuracy:** 'Bird' class (Acc: 41%, Entropy: 1.27).

### Task 3: Behavior on Unseen Images (OOD)

#### A. CIFAR-100 (Unseen Semantic Classes)
We fed CIFAR-100 images into the CIFAR-10 model to observe feature mapping.
* **Observation:** The model mapped visually similar classes to known categories with high confidence.
    * *Example:* 'Pickup Truck' (CIFAR-100) $\rightarrow$ Classified confidently as 'Car' or 'Truck'.
    * *Example:* 'Snake'/'Dinosaur' $\rightarrow$ Resulted in high entropy (low confidence).

#### B. Gaussian Noise Analysis
We injected pure Gaussian noise with varying means and standard deviations to test feature robustness.
* **Result:** The model displayed high confidence (low entropy) on pure noise.
* **Bias:** High-frequency noise was overwhelmingly classified as **"Car"**.
* **Conclusion:** The model learned to associate high-texture inputs with the 'Car' class rather than learning high-level shape semantics, demonstrating a lack of robustness.

## Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/cifar10-robustness-analysis.git](https://github.com/yourusername/cifar10-robustness-analysis.git)
    cd cifar10-robustness-analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib numpy
    ```

3.  **Run the Notebook:**
    Open `cifar10_tutorial.ipynb` in Jupyter Lab, Jupyter Notebook, or Google Colab to reproduce the training and experiments.
