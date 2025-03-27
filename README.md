# Neural Network Architectures: Single-Layer vs Deep MLPs

This project compares the performance of a neural network with one hidden layer and a deep neural network (DNN) using the CelebA and MNIST datasets. The objective is to analyze how architectural depth and regularization impact model accuracy and training efficiency.

## Project Overview

This repository includes:

- Skeleton implementations of neural networks in `.py` files for both CelebA and MNIST.
- A consolidated Jupyter notebook `ML - PROJECT 1.ipynb` where all experiments, visualizations, and evaluations are conducted.
- Feature selection, hyperparameter tuning, model comparison, and result analysis.

## File Structure

```
├── face_nn.py               # Single-layer NN (CelebA) - logic only
├── dnn.py                   # Deep NN (CelebA, PyTorch) - logic only
├── nn_script.py             # Single-layer NN (MNIST) - logic only
├── ML - PROJECT 1.ipynb     # Main notebook with all runs and outputs
├── face_all.pickle          # CelebA dataset
├── mnist_all.mat            # MNIST dataset
├── *.pkl                    # Model parameters and results
├── *.png                    # Output plots
└── README.md
```

## Datasets

- **CelebA**: Binary classification (wearing glasses or not), with pre-extracted features.
- **MNIST**: Digit classification (0–9), loaded from MATLAB `.mat` file.

## Models Compared

### Single-Layer Neural Network

- Implemented in NumPy & SciPy (`face_nn.py`, `nn_script.py`)
- Optimization using Conjugate Gradient
- Hyperparameter tuning over hidden units and regularization (lambda)

### Deep Neural Network (DNN)

- Built using PyTorch (`dnn.py`)
- Configurable number of layers: 4, 12, and 20
- Uses ReLU, BatchNorm, Dropout
- Trained with Adam optimizer

## Results Overview

| Model           | Dataset | Best Accuracy | Training Time | Best Config               |
|----------------|---------|---------------|----------------|---------------------------|
| face_nn        | CelebA  | 85.29%        | 44.06 sec      | 1 hidden unit, λ = 12     |
| dnn            | CelebA  | 86.41%        | 82.74 sec      | 12 layers, LR = 0.01      |
| nn_script      | MNIST   | 93.57%        | 69.94 sec      | 20 hidden units, λ = 1    |

## Visualizations

Included in the notebook:

- Accuracy vs Regularization (λ)
- Accuracy vs Hidden Units
- Accuracy vs Number of Layers (DNN)
- Training Time vs Number of Layers (DNN)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nn-vs-dnn-comparison
   cd nn-vs-dnn-comparison
   ```

2. Install the required packages:
   ```bash
   pip install numpy scipy matplotlib torch
   ```

3. Open the notebook:
   ```bash
   jupyter notebook "ML - PROJECT 1.ipynb"
   ```

> Note: The `.py` files serve as modular logic, but **all execution, plots, tuning, and comparisons are done in the notebook**.
