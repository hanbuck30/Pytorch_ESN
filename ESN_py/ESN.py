import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils.sparseLIB import sparseESN
from .utils.func import check_dim, check_type, metric_func
from .learning_algorithm.Gradient_descent import gd_init, gd_rule, batch_gd_rule
from .learning_algorithm.Online_learning import Online_init, Online_rule
from .learning_algorithm.Inverse_matrix import inverse_init, inverse_rule

from tqdm.notebook import tqdm



class ESN(sparseESN):
    def __init__(self, args):
        super(ESN, self).__init__(args)
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        
    def fit(self, input_source=None, output=None, test_x=None, test_y=None, test_x_type="float", test_y_type="float", dataloader=None, test_dataloader=None):
        if dataloader is None:
            # Check dimensions and types of input and output
            input_source, output = check_dim(input_source, output)
            input_source, Yt = check_type(self, input_source, output, test_x_type, test_y_type)

            n_input, n_f = input_source.shape
 
            if self.Test:
                test_y = check_type(self, output=test_y, output_type=test_y_type)[1]
        
        if self.l_a == "gd":
            criterion, optimizer = gd_init(self)
            # Training loop
            for epoch in tqdm(range(self.epoch)):
                if dataloader is None:
                    gd_rule(self, Yt, input_source, n_input, criterion, optimizer, epoch)
                    if self.Test:
                        predict_result = self.predict(test_x)
                    metric_func(predict_result, test_y)
                elif dataloader is not None:
                    batch_gd_rule(self, dataloader, criterion, optimizer, epoch)
                    if self.Test:
                        predict_result = self.batch_predict(test_dataloader)
                        metric_func(predict_result.view(-1,test_y.shape[1]), test_y.view(-1,test_y.shape[1]))



        elif self.l_a == "online":
            Online_init(self)
            Online_rule(self, input_source, n_input, Yt)
            if self.Test:
                predict_result = self.predict(test_x)
                metric_func(predict_result, test_y)

        elif self.l_a == "inverse":
            inverse_init(self,n_input)
            inverse_rule(self, input_source, n_input, Yt)
            if self.Test:
                predict_result = self.predict(test_x)
                metric_func(predict_result, test_y)

            

    def predict(self, input_source, input_type="float"):
        # Check dimensions and types of input
        input_source = check_dim(input_source)[0]
        n_input, n_f = input_source.shape
        input_source = check_type(self, input_source=input_source, input_type=input_type)[0]
        
        # Initialize reservoir state
        self.x = torch.zeros((1, self.resSize), device=self.device)
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
    
    def batch_predict(self, dataloader):
        
        
        all_pred = None
        for i, sample in enumerate(dataloader):
            test_source, Yt = sample
            test_source = test_source.to(self.device)
            Yt = Yt.to(self.device) 
            predicted_val = torch.zeros_like(Yt, dtype=torch.float, device=self.device, requires_grad=False) # prediction data

            batch_size, n_input, input_feature = test_source.shape
            self.batch_x = torch.zeros((batch_size, self.resSize), device=self.device)
            _, n_input, output_feature = Yt.shape

            for t in range(n_input):
                
                # Making input tensor
                u = test_source[:,t,:].reshape(batch_size, -1)
                # Update reservoir state
                self.batch_x = self.batch_update_state(u)
                extended_state = torch.hstack([self.ones, u, self.batch_x])
                # Forward pass
                prediction = torch.matmul(extended_state, self.Wout)
                predicted_val[:,t,:] = prediction
        
            if all_pred is None:
                all_pred = predicted_val
            else:
                all_pred=torch.concat((all_pred,predicted_val), axis = 0)

        return all_pred
    

    def generate(self, u, generate_len, u_type = "float"):
        # Initialize reservoir state
        u = check_dim(u)[0]
        predicted_val = torch.zeros(generate_len, self.output_dim)
        u = check_type(self, input_source=u, input_type=u_type)[0]

        for t in range(generate_len):
            self.x = self._update_state(u)
            extended_state = torch.hstack([self.one, u, self.x])

            # Forward pass
            prediction = torch.matmul(extended_state, self.Wout).reshape(1, -1)
            predicted_val[t, :] = prediction
            u = prediction

        return predicted_val
