import torch
import numpy as np
import random

def get_loss_func(args):
    if args.Loss_function == 'mse':
        return torch.nn.MSELoss()
    
    elif args.Loss_function == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    
    elif args.Loss_function == 'log_loss':
        return torch.nn.NLLLoss()

def get_torch_func(args): # 파라미터 inter_unit 데이터 타입을 맞춰주기 위해서 만든 함수입니다.
    if args.inter_unit == 'tanh':
        return torch.tanh
    elif args.inter_unit == 'sin':
        return torch.sin
    elif args.inter_unit == 'relu':
        return  torch.relu
    elif args.inter_unit == 'leaky_relu':
        return  torch.leaky_relu
    elif args.inter_unit == 'sigmoid':
        return  torch.sigmoid
    else:
        raise ValueError('Invalid function name')

def set_common(self, args):
    self.resSize = int(args.resSize)
    self.n_feature = args.input_dim
    self.d_model = args.d_model
    self.output_dim = args.output_dim
    self.damping = np.around(args.damping,5)
    self.spectral_radius = np.around(args.spectral_radius,5)
    self.weight_scaling = np.around(args.weight_scaling,5)
    self.input_scaling = np.around(args.input_scaling,5)
    self.l_a = args.l_a
    self.weight_sparsity = np.around(args.weight_sparsity,5)
    self.input_sparsity = np.around(args.input_sparsity,5)
    self.Loss_function = get_loss_func(args)
    self.Test = args.Test
    self.task = args.task
    
    inter_unit1 = get_torch_func(args)
    self.inter_unit = inter_unit1
    
def set_gd(self, args):
    self.learning_rate = args.learning_rate
    self.epoch = args.epoch
    self.l2_lambda = args.l2_lambda # L2 규제 계수

def set_online(self, args):
    self.lambda_ = args.lambda_
    self.n_gamma = args.n_gamma 

def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return