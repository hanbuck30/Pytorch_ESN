import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ESN_py.utils.sparseLIB import sparseESN
from ESN_py.utils.func import check_dim, check_type, metric_func
from ESN_py.config import parse_args

from tqdm.notebook import tqdm
import argparse
import json
import logging
import os


class ESN(sparseESN):
    def __init__(self, args):
        super(ESN, self).__init__(args)
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        
    def fit(self, input_source, output, test_x=None, test_y=None):
        # Check dimensions and types of input and output
        input_source, output = check_dim(input_source, output)
        input_source, Yt = check_type(self, input_source, output, "float", "float")

        n_input, n_f = input_source.shape
        
        if self.Test:
            test_y = check_type(self, output=test_y, output_type="float")[1]

        # Define the loss function and optimizer
        criterion = self.Loss_function
        parameters = [self.Wout]
        optimizer = optim.Adam(parameters, self.learning_rate)

        # Training loop
        for i in tqdm(range(self.epoch)):
            Y = torch.zeros_like(Yt, dtype=torch.float, device=self.device, requires_grad=False) # prediction data
            self.x = torch.zeros((1, self.resSize), device=self.device)
            
            for t in range(n_input):
                # Making input tensor
                u = input_source[t, :].reshape(1, -1)
                # Update reservoir state
                self.x = self._update_state(u)
                extended_state = torch.hstack([self.one, u, self.x])
                # Forward pass
                Y[t, :] = torch.matmul(extended_state, self.Wout).reshape(-1)

            # Compute the loss
            loss = criterion(Y, Yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the training loss
            print("epoch :{}/{}  Loss : {:.5f}".format(i, self.epoch, loss.item()))
            
            if self.Test:
                predict_result = self.predict(test_x)
                metric_func(predict_result, test_y)

    def predict(self, input_source):
        # Check dimensions and types of input
        input_source = check_dim(input_source)[0]
        n_input, n_f = input_source.shape
        input_source = check_type(self, input_source=input_source, input_type="float")[0]
        
        # Initialize reservoir state
        x = torch.zeros((1, self.resSize), device=self.device)
        predicted_val = torch.zeros(n_input, self.output_dim)

        for t in range(n_input):
            # Making input tensor
            u = input_source[t, :].reshape(1, -1).type(torch.float).to(self.device)
            # Update reservoir state
            x = self._update_state(u)
            extended_state = torch.hstack([self.one, u, x])
            # Forward pass
            prediction = torch.matmul(extended_state, self.Wout).reshape(1, -1)
            predicted_val[t, :] = prediction

        return predicted_val
