# Domain Adaptation Experiment

## Project Overview

This project focuses on fine-tuning deep learning models for image classification. It includes experiments on:

- **MNIST:** Handwritten digit recognition.
- **SVHN:** Street View House Numbers classification.
- **Domain Adaptation:** Transferring learned features between MNIST and SVHN.


## Setup Instructions

### Prerequisites

- Python 3.7+
- PyTorch
- Jupyter Notebook


### Installation Steps

1. **Clone the repository**

```bash
git clone <your-repo-url>
```

2. **Navigate to the project directory**

```bash
cd fine_tuning_exp
```


## Datasets

### MNIST

- **Description:** Handwritten digit classification (0–9).
- **Dataset Size:** 60,000 training images; 10,000 test images.
- **Image Format:** 28×28 grayscale.
- **Download:**
Download the dataset from [Kaggle: MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and extract it to:

```
data/MNIST/
```


### SVHN

- **Description:** Street View House Numbers (real-world digit images).
- **Image Format:** 32×32 color images.
- **Download:**
Download the dataset from [Stanford: SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) and extract it to:

```
data/SVHN/
```


## Running Experiments

1. **Train on MNIST**
    - Run the training script or Jupyter notebook for MNIST.
    - A `.pth` (model checkpoint) file will be saved upon completion.
2. **Train on SVHN**
    - Run the SVHN training script independently to observe baseline results.
3. **Domain Adaptation**
    - Use the saved MNIST model to fine-tune on SVHN data for domain adaptation experiments.

> **Note:**
> - Ensure you download the datasets and place them in the specified folders before starting the experiments.
> - Adjust any data-loading paths in your scripts if required for your environment.

