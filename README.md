# Pytorch_ESN
ESN version of pytorch
# Echo State Network (ESN) Implementation

This repository contains an implementation of the Echo State Network (ESN) algorithm along with an example usage notebook and utility functions.

## Repository Structure

- `ESN_example.ipynb`: Jupyter notebook providing a usage example of the ESN implementation.
- `ESN_py/`: Core package directory containing the implementation of the ESN.
  - `__init__.py`: Initialization file for the ESN_py package.
  - `config.py`: Configuration file for the ESN parameters.
  - `ESN.py`: Core implementation of the Echo State Network.
  - `learning_algorithm/`: Directory containing different learning algorithms for the ESN.
    - `Gradient_descent.py`: Implementation of gradient descent learning algorithm.
    - `Inverse_matrix.py`: Implementation of inverse matrix learning algorithm.
    - `Online_learning.py`: Implementation of online learning algorithm.
  - `utils/`: Directory containing utility functions and settings for the ESN.
    - `func.py`: Utility functions for the ESN.
    - `setting.py`: Settings and configurations for the ESN utilities.
    - `sparseLIB.py`: Sparse matrix operations library.
- `MackeyGlass_t17.txt`: Dataset used in the example notebook.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed. Install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```
### Installation

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/hanbuck30/Pytorch_ESN.git
```

