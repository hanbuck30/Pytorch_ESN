import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .setting import set_random_seed, set_common, set_gd, set_online

class sparseESN(nn.Module):
    def __init__(self, args):
        super(sparseESN, self).__init__()

        USE_CUDA = torch.cuda.is_available()
        device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        set_random_seed(0)
        set_common(self, args)
        set_gd(self, args)
        set_online(self, args)

        # Initialize constant tensor with value 1
        self.one = torch.tensor([[1.0]], device=device)
        
        # Initialize input weights
        W_in = self._create_sparse_reservoir(1 + self.n_feature, self.resSize, self.input_sparsity, self.spectral_radius, "Win", self.input_scaling, self.weight_scaling)
        self.Win = nn.Parameter(W_in, requires_grad=False).to(device)
        
        # Create sparse reservoir
        W_data = self._create_sparse_reservoir(self.resSize, self.resSize, self.weight_sparsity, self.spectral_radius, "W", self.input_scaling, self.weight_scaling)
        self.W = nn.Parameter(W_data, requires_grad=False).to(device)
        
        # Initialize output weights
        self.Wout = nn.Parameter(torch.zeros(1 + self.n_feature + self.resSize, self.output_dim, requires_grad=True).to(device))

        # Initial state (not a Parameter, since we won't be updating it via traditional backpropagation)
        self.x = torch.zeros((1, self.resSize)).to(device)


    def _create_sparse_reservoir(self, row_size, col_size, sparsity, spectral_radius, typ, input_scaling, weight_scaling):
        # Create sparse matrix with specified sparsity, then convert to PyTorch dense tensor for simplicity
        # Define the size and desired sparsity

        # Generate random indices for the non-zero entries
        row_indices = torch.randint(0, row_size, (1, int(row_size * col_size * sparsity)))
        col_indices = torch.randint(0, col_size, (1, int(row_size * col_size * sparsity)))
        indices = torch.cat((row_indices, col_indices), dim=0)

        # Generate random values between -1 and 1 for the non-zero entries
        values = 2 * torch.rand(indices.shape[1]) - 1
        
        # Create the sparse tensor using the COO format
        reservoir = torch.sparse_coo_tensor(indices, values, size=(row_size, col_size))

        # Adjust the spectral radius for the reservoir matrix W
        if (spectral_radius == 0.0) and (typ == "W"):
            # Make symmetric matrix (if needed)
            reservoir_dense = reservoir.to_dense()
            
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvalsh(reservoir_dense)
            spectral_radius = torch.max(torch.abs(eigenvalues))
            reservoir = (weight_scaling / spectral_radius) * reservoir
        elif (spectral_radius != 0.0) and (typ == "W"):
            reservoir = (weight_scaling / spectral_radius) * reservoir
        elif typ == "Win":
            reservoir = input_scaling * reservoir
        
        return reservoir
    
    def _update_state(self, u):
        # Update the reservoir state
        self.x = (1 - self.damping) * self.x + self.damping * self.inter_unit(torch.matmul(torch.hstack([self.one, u]), self.Win) + torch.matmul(self.x, self.W))
        return self.x
    
    def batch_update_state(self, u):
        # Update the reservoir state
        self.ones = torch.concat([self.one]*u.shape[0])
        self.batch_x = (1 - self.damping) * self.batch_x + self.damping * self.inter_unit(torch.matmul(torch.hstack([self.ones, u]), self.Win) + torch.matmul(self.batch_x, self.W))
        return self.batch_x
