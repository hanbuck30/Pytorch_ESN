import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    print("extracting arguments")

    parser.add_argument("--resSize", type=int, default=3000)
    parser.add_argument("--n_feature", type=int, default=1)
    parser.add_argument("--damping", type=float, default=0.3)
    parser.add_argument("--spectral_radius", type=float, default=0.1)
    parser.add_argument("--weight_scaling", type=float, default=1.25)
    parser.add_argument("--input_scaling", type=float, default=1.25)
    parser.add_argument("--output_dim", type=int, default=1) # regression 일때, n_label 신경 X
    parser.add_argument("--inter_unit", type=str, default='tanh')

    parser.add_argument("--learning_rate", type=float, default=1e-1) # Gradient Descent 알고리즘
    parser.add_argument("--l_a", type=str, default='gd') # Learning 알고리즘
    parser.add_argument("--sparsity", type=float, default=0.01)

    parser.add_argument("--l2_lambda", type=float, default=0.0)
    parser.add_argument("--epoch", type=int, default = 100)
    parser.add_argument("--Loss_function", type=str, default = "mse")
    parser.add_argument("--Test", type=bool, default = True)


    args, _ = parser.parse_known_args()

    return args