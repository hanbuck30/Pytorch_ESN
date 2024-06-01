# Pytorch_ESN
ESN version of pytorch
# Echo State Network (ESN) Implementation

This repository contains an implementation of the Echo State Network (ESN) and a usage example. The ESN is a type of recurrent neural network used for time-series prediction tasks. This implementation includes the necessary configuration, utility functions, and an example notebook to demonstrate how to use the ESN.

## Repository Structure

- `ESN_example.ipynb`: Jupyter notebook providing a usage example of the ESN implementation.
- `ESN_py/`
  - `ESN.py`: Core implementation of the Echo State Network.
  - `__init__.py`: Initialization file for the ESN_py package.
  - `config.py`: Configuration file for the ESN parameters.
  - `utils/`
    - `func.py`: Utility functions for the ESN.
    - `setting.py`: Settings and configurations for the ESN utilities.
    - `sparseLIB.py`: Sparse matrix operations library.
- `MackeyGlass_t17.txt`: Dataset used in the example notebook.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed. Install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
