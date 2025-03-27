# Neural Network Architectures: Single-Layer vs Deep MLPs

This project compares the performance of a neural network with one hidden layer and a deep neural network (DNN) using the CelebA and MNIST datasets. The goal is to evaluate how network depth and regularization impact model accuracy and training efficiency.

## Project Overview

This repository includes:

- A NumPy-based single hidden layer neural network for both CelebA and MNIST datasets.
- A PyTorch-based deep neural network (DNN) with configurable layer depth.
- Plots and results for accuracy vs hidden units, regularization, training time, and model comparison.
- Feature selection and hyperparameter tuning for regularization (lambda) and learning rate.

## File Structure

```
├── face_nn.py               # Single hidden layer NN on CelebA (NumPy)
├── dnn.py                   # Deep Neural Network using PyTorch (CelebA)
├── nn_script.py             # Single hidden layer NN on MNIST (NumPy)
├── face_all.pickle          # CelebA features and labels
├── mnist_all.mat            # Raw MNIST dataset (MAT format)
├── *.pkl                    # Pickled model parameters and evaluation results
├── *.png                    # Accuracy/training time plots
└── README.md
```

## Datasets

- **CelebA**: Binary classification (wearing glasses or not), with preprocessed features.
- **MNIST**: Digit classification (0 to 9), loaded from MATLAB `.mat` file.

## Models

### Single-Layer Neural Network (face_nn.py, nn_script.py)

- Developed using NumPy and SciPy
- Trained using the Conjugate Gradient optimization algorithm
- Tested on multiple configurations of hidden units and regularization values
- Includes feature selection and saving of best parameters

### Deep Neural Network (dnn.py)

- Built with PyTorch
- Configurable depth: 4, 12, and 20 hidden layers
- Uses Batch Normalization, ReLU, and Dropout
- Tuned for different learning rates: 0.5, 0.1, 0.01

## Results Summary

| Model              | Dataset | Best Accuracy | Training Time | Best Configuration        |
|-------------------|---------|---------------|----------------|----------------------------|
| face_nn.py        | CelebA  | 85.29%        | 44.06 sec      | 1 hidden unit, lambda=12   |
| dnn.py            | CelebA  | 86.41%        | 82.74 sec      | 12 layers, LR = 0.01       |
| nn_script.py      | MNIST   | 93.57%        | 69.94 sec      | 20 hidden units, lambda=1  |

## Visualizations

Several plots were generated to compare:

- Validation Accuracy vs. Regularization (lambda)
- Validation Accuracy vs. Number of Hidden Units
- Validation Accuracy vs. Number of Layers (DNN)
- Training Time vs. Number of Layers (DNN)

These are included in the output section and referenced in the final report.

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nn-vs-dnn-comparison
   cd nn-vs-dnn-comparison
   ```

2. Install required packages:
   ```
   pip install numpy scipy matplotlib torch
   ```

3. Run the scripts:
   ```
   python face_nn.py
   python dnn.py
   python nn_script.py
   ```

## Notes

- Preprocessing includes normalization and variance-based feature selection.
- All scripts save their best-performing model parameters and selected features in `.pickle` files.
- Training logs and output plots help support the analysis and comparison.
