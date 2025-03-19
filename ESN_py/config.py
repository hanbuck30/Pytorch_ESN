import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    print("extracting arguments")
    ## Model Settings
    parser.add_argument("--resSize", type=int, default=15000)
    parser.add_argument("--input_dim", type=int, default=122)
    parser.add_argument("--output_dim", type=int, default=80)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--spectral_radius", type=float, default=0.1)
    parser.add_argument("--weight_scaling", type=float, default=1.0)
    parser.add_argument("--input_scaling", type=float, default=1.0)
    parser.add_argument("--inter_unit", type=str, default='tanh')
    parser.add_argument("--weight_sparsity", type=float, default=0.001)
    parser.add_argument("--input_sparsity", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default= 64)
    
    # Learning Algorithm
    parser.add_argument("--l_a", type=str, default='gd') 

    ## GD Learning
    parser.add_argument("--learning_rate", type=float, default=2e-3) # Gradient Descent Algorithm
    parser.add_argument("--l2_lambda", type=float, default=0.0001)
    parser.add_argument("--epoch", type=int, default = 100)
    parser.add_argument("--Loss_function", type=str, default = "mse")

    ## Online Learning
    parser.add_argument("--lambda_", type=float, default=1.0) # RLS Algorithm
    parser.add_argument("--n_gamma", type=float, default=2e-4) # Force Algorithm

    ## With or without testing at the same time as learning
    parser.add_argument("--Test", type=bool, default = True)

    ## Classification and regression task possible
    parser.add_argument("--task", type=str, default = 'cls')

    args, _ = parser.parse_known_args()

    return args