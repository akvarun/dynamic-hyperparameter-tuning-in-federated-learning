# Dynamic Hyperparameter Tuning in Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

Federated learning pipeline with dynamic hyperparameter optimization using SMAC (Sequential Model-based Algorithm Configuration). Accelerates ML model convergence across distributed clients with GPU acceleration support for Linux and Mac environments.

Demonstrates performance evaluation on CIFAR-10 datasets and SVM models with efficient hyperparameter search in distributed ML workflows.

## Features

- GPU-accelerated hyperparameter tuning using SMAC
- Federated learning across multiple clients  
- Reproducible environment setup for Linux and MacOS
- Performance evaluation on CIFAR-10 and SVM classifiers
- 3× faster hyperparameter search with stable model accuracy
- Optimized for edge and distributed systems workflows

## Installation

### Linux Setup
```bash
conda create -n SMAC python=3.10
conda activate SMAC
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac
```

### MacOS Setup
```bash
conda create -n SMAC python=3.10
conda activate SMAC
pip install smac
brew install swig
```

## Usage

1. Place CIFAR-10 dataset in `./data` directory
2. Run optimization scripts:
   ```bash
   python minimal_example.py
   python minimal_optimize_static_lr_svm.py
   ```
3. Monitor performance metrics and convergence rates

## Technical Stack

- **Hyperparameter Optimization**: SMAC
- **Deep Learning**: PyTorch with GPU acceleration
- **Datasets**: CIFAR-10, SVM models
- **Environment**: Linux/MacOS with Conda

## Results

- **3× faster** hyperparameter search compared to static methods
- Stable model accuracy across distributed clients
- Improved convergence rates with dynamic tuning

## License

MIT License - see [LICENSE](LICENSE) file for details.
