import torch
import torch.nn as nn
from tqdm.notebook import tqdm


def inverse_init(self, n_input):    
    self.X = torch.zeros((n_input, 1+self.n_feature + self.resSize),dtype = torch.float, device=self.device)

def inverse_rule(self, input_source, n_input, Yt):
    for t in tqdm(range(n_input)):
        u = input_source[t,:].reshape(1,-1) # input에서 값을 하나씩 들고온다
        self.x = self._update_state(u)
        # x에 전체노드에서 소실률에 의거해 위의 식에 따라 계산된 weight값을 저장한다
        self.X[t, :] = torch.hstack([self.one,u,self.x])[0,: ]  # X에 1,u,x를 쌓아 저장한다

    #### train the output by ridge regression
    # reg = 1e-8  # regularization coefficient
    #### direct equations from texts:
    # X_T = X.T
    # Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
    # reg*np.eye(1+inSize+resSize) ) )
    # using scipy.linalg.solve:
    # inverse matrix를 이용한 ridge regression을 이용하여 Wout을 구할 수 있음
    reg = 1e-8
    self.Wout = nn.Parameter(torch.linalg.solve(torch.matmul(self.X.T,self.X) + reg*torch.eye(1+self.n_feature+self.resSize).to(self.device), torch.matmul(Yt.T,self.X).T))