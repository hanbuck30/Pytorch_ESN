import torch
from tqdm.notebook import tqdm

def Online_init(self):
    self.P = (1.0 / self.lambda_) * torch.eye(self.resSize + self.n_feature + 1).to(self.device)
    return
def Online_rule(self, input_source, n_input, Yt):        
    for t in tqdm(range(n_input)):
        u = input_source[t,:].reshape(1,-1).type(torch.float).to(self.device)  
        # Update reservoir state
        self.x = self._update_state(u)
        extended_state = torch.hstack([self.one, u, self.x])

        # RLS Update
        pt = self.P @ extended_state.T
        gamma = 1.0 / (self.lambda_ + extended_state @ pt )
        if self.n_gamma != 0: # n_gamma가 0일때, RLS 알고리즘
            gamma = self.n_gamma
        self.P.sub_(gamma * pt @ pt.T)  # 인플레이스 연산으로 변경된 부분
        self.Wout.requires_grad = False
        if (pt * gamma * (Yt[t,:].reshape(1, -1) - extended_state @ self.Wout)).ndim >=3:
            self.Wout += (pt * gamma * (Yt[t,:].reshape(1, -1) - extended_state @ self.Wout)).squeeze()
        else:
            self.Wout += (pt * gamma * (Yt[t,:].reshape(1, -1) - extended_state @ self.Wout)) 
        del u, gamma, extended_state, pt
    return 