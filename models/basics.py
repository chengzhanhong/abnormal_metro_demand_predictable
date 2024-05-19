import torch
from torch import nn
import numpy as np

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high   
        # self.low, self.high = ranges        
    def forward(self, x):                    
        # return sigmoid_range(x, self.low, self.high)
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm2d(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

class Standardization(nn.Module):
    def __init__(self, loc, scale):
        super(Standardization, self).__init__()
        self.loc = loc.reshape([-1, 1, 1])
        self.scale = scale.reshape([-1, 1, 1])

    def forward(self, x, i, mode: str):
        """
        x: (bs, num_patch, output_len)
        i: index of the patch, (bs,)
        """
        if mode == 'norm':
            x = (x - self.loc[i]) / self.scale[i]
        elif mode == 'denorm':
            x = x * self.scale[i] + self.loc[i]
        else:
            raise NotImplementedError
        return x

class RmseHead(nn.Module):
    def __init__(self, d_model, output_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, output_len)

    def forward(self, x):
        x = self.linear(self.dropout(x))
        return x

class LogNormalHead(nn.Module):
    """Produces parameters for a log-normal distribution."""
    def __init__(self, d_model, output_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mu = nn.Linear(d_model, output_len)
        self.sigma = nn.Linear(d_model, output_len)
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: [bs x num_patch x d_model]
        mu = self.mu(self.dropout(x))
        sigma = self.sigma(self.dropout(x))
        sigma = self.softplus(sigma).clamp(min=1e-8)  # [bs x num_patch x n_components]
        return mu, sigma

class HeadNegativeBinomial(nn.Module):
    def __init__(self, d_model, output_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_success = nn.Linear(d_model, output_len)
        self.p_success = nn.Linear(d_model, output_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        n_success = self.softplus(self.n_success(self.dropout(x)))
        p_success = self.sigmoid(self.p_success(self.dropout(x)))
        return n_success, p_success


#%% Negative log likelihood
def nll_NB(n_success, p_success, y):
    """Compute the negative log likelihood of Negative Binomial distribution, element-wise."""
    nll = (- torch.lgamma(n_success + y) + torch.lgamma(n_success) +
           torch.lgamma(y + 1) - n_success * torch.log(p_success) -
           y * torch.log(1 - p_success))
    return nll

def nll_LogNormal(mu, sigma, y):
    """Compute the negative log likelihood of LogNormal distribution, element-wise."""
    nnl = -torch.log(sigma) - 0.5 * torch.log(2 * np.pi) - 0.5 * ((torch.log(y) - mu) / sigma) ** 2 - torch.log(y)
    return nnl


#%% Sampling functions
def sampleNB(n_success, p_success):
    total_count = n_success
    probs = 1 - p_success
    return torch.distributions.NegativeBinomial(total_count, probs).sample((1,)).squeeze(0)

def sampleLogNormal(mu, sigma):
    dist = torch.distributions.LogNormal(mu, sigma)
    return dist.sample((1,)).squeeze(0)

def sampleRMSE(mu):
    return mu

#%% Mean function
def meanRMSE(mu):
    return mu

def meanNB(n_success, p_success):
    return n_success * (1 - p_success) / p_success

def meanlogNormal(mu, sigma):
    return torch.exp(mu + sigma ** 2 / 2)


#%% aggregations
head_dic = {'NB': HeadNegativeBinomial, 'logNormal': LogNormalHead, 'RMSE': RmseHead}

mean_dic = {'NB': meanNB, 'logNormal': meanlogNormal, 'RMSE': meanRMSE}

sample_dic = {'NB': sampleNB, 'logNormal': sampleLogNormal, 'RMSE': sampleRMSE}

