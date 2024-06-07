import torch
import torch.optim as optim
import time
from tqdm.notebook import tqdm

def gd_init(self):
    # Define the loss function and optimizer
    criterion = self.Loss_function
    parameters = [self.Wout]
    optimizer = optim.Adam(parameters, self.learning_rate)
    return criterion, optimizer

def gd_rule(self, Yt, input_source, n_input, criterion, optimizer, epoch):
    start_time = time.time()
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

    # Calculate timestamp
    end_time = time.time()
    time_required = end_time - start_time

    # Print the training loss
    print("epoch :{}/{}  Loss : {:.5f}  time_required : {}".format(epoch, self.epoch, loss.item(), time_required))
    return 



