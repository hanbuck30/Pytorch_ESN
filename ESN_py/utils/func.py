import torch
import numpy as np
import gc
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score

def check_dim(input_source, output=None):
    # Check and reshape input to 2D if necessary
    if input_source.ndim == 2:
        pass
    elif input_source.ndim == 1:
        input_source = input_source.reshape(-1, 1)
    elif input_source.ndim == 0:
        input_source = input_source.reshape(-1, 1)
        # print("It needs 2-dimensions, so we're going to change the dimensions")

    # Check and reshape output to 2D if necessary
    if output is None or output.ndim == 2:
        pass
    elif output.ndim == 1:
        output = output.reshape(-1, 1)
    elif output.ndim == 0:
        output = output.reshape(-1, 1)
        # print("It needs 2-dimensions, so we're going to change the dimensions")
    return input_source, output

def check_type(self, input_source=None, output=None, input_type=None, output_type=None):
    # Convert input to specified type and move to device if necessary
    if input_source is None and input_type is None:
        pass
    elif type(input_source) != torch.Tensor:
        input_source = input_source.astype(input_type)
        torch_inp_type = convert_torch_type(input_type)
        input_source = torch.tensor(input_source, dtype=torch_inp_type, device=self.device, requires_grad=False)
    
    # Convert output to specified type and move to device if necessary
    if output is None and output_type is None:
        pass
    elif type(output) != torch.Tensor:
        output = output.astype(output_type)
        torch_oup_type = convert_torch_type(output_type)
        output = torch.tensor(output, device=self.device, dtype=torch_oup_type)

    return input_source, output

def clear_cash(t):
    # Clear cache if needed
    if torch.cuda.is_available() and t % 100000 == 0:
        torch.cuda.empty_cache()
        gc.collect()

def convert_torch_type(ori_type):
    # Convert original type to corresponding PyTorch type
    if ori_type == 'float':
        torch_inp_type = torch.float32
    elif ori_type == 'int':
        torch_inp_type = torch.int
    elif ori_type == 'double':
        torch_inp_type = torch.float64
    return torch_inp_type

def metric_func(pred, target):

    if type(target) == np.ndarray:
        tar_cpu = target
    elif type(target) == torch.Tensor:
        tar_cpu = target.detach().cpu().numpy()
    if type(pred) == np.ndarray:
        pre_cpu = pred
    elif type(pred)  == torch.Tensor:
        pre_cpu = pred.detach().cpu().numpy()
    

    mse = mean_squared_error(tar_cpu, pre_cpu)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(tar_cpu, pre_cpu)
    r2 = r2_score(tar_cpu, pre_cpu)

    # Print the test metric
    print("Test Metric MSE: {:.5f}   RMSE: {:.5f}   MAE: {:.5f}   r2score: {:.5f}".format(mse, rmse, mae, r2))
    return 
