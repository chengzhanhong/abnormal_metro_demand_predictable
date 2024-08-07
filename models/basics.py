import torch
from torch import nn
import numpy as np
import time
import datetime
import wandb
import matplotlib.pyplot as plt

class FlattenLastTwoDims(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.flatten(x, start_dim=-2)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


def get_quantile_edges(data, num_bins, max_leap_rate=0.1, plot=False):
    quantile_edges = np.linspace(0, 1, num_bins)
    bin_edges = np.quantile(data, quantile_edges)
    leaps = np.diff(bin_edges)
    idx, max_leap = np.argmax(leaps), np.max(leaps)

    while max_leap > max_leap_rate * bin_edges[idx+1]:
        quantile_edges = np.insert(quantile_edges, idx+1, (quantile_edges[idx] + quantile_edges[idx+1]) / 2)
        bin_edges = np.quantile(data, quantile_edges)
        leaps = np.diff(bin_edges)
        idx, max_leap = np.argmax(leaps), np.max(leaps)
    bin_edges, idx = np.unique(bin_edges, return_index=True)
    bin_edges = np.sort(bin_edges)
    bin_edges[-1] = bin_edges[-1] + 1e-6 # to include the last value
    quantile_edges = quantile_edges[idx]
    # get the number of samples in each bin
    num_per_bin = np.histogram(data, bins=bin_edges)[0]

    if plot:
        fig, ax = plt.subplots()
        ax.hist(data, bins=num_bins)
        fig, ax = plt.subplots()
        ax.hist(data, bins=bin_edges)
    return bin_edges, num_per_bin, quantile_edges


def num2bin(x, bin_edges):
    # Here, the x outsides of the bin_edges will be assigned to the last bin
    if isinstance(x, torch.Tensor):
        bin = torch.clamp(torch.bucketize(x, bin_edges, right=True) - 1, min=0, max=len(bin_edges)-2)
    else:
        bin = np.minimum(np.digitize(x, bin_edges) - 1, len(bin_edges)-2)
    return bin


def bin2num(x, bin_edges):
    bin_len = len(bin_edges) -1
    if isinstance(x, torch.Tensor):
        return 0.5 * (bin_edges[x] + bin_edges[torch.minimum(x+1, torch.tensor(bin_len))]-1).to(torch.float32)
    else:
        return 0.5 * (bin_edges[x] + bin_edges[np.minimum(x+1, bin_len)]-1)


class Standardization(nn.Module):
    def __init__(self, loc, scale, input_type='number', head='RMSE'):
        """
        input_type: 'number' or 'bins'
        head: 'RMSE' or 'NB' or 'logNormal', CrossEntropy
        """
        super(Standardization, self).__init__()
        self.input_type = input_type
        self.head = head
        self.softplus = nn.Softplus()
        self.loc = loc.reshape([-1, 1, 1])
        self.scale = scale.reshape([-1, 1, 1])

    def forward(self, x, i, mode: str):
        """
        x: (bs, num_patch, output_len)
        i: index of the station
        """
        if mode == 'norm':
            if self.input_type == 'number':
                x = (x - self.loc[i]) / self.scale[i]
            elif self.input_type == 'bins':
                pass
            else:
                raise ValueError('input_type not supported, should be "number" or "bins"')

        elif mode == 'denorm':
            if self.head == 'RMSE':
                x[0][:,:,:] = self.softplus(x[0] * self.scale[i] + self.loc[i])
            elif self.head == 'NB':
                x[0][:,:,:] = x[0] * self.scale[i]  # n_success should not add loc
            elif self.head in {'Normal', 'TruncatedNormal'}:
                x[0][...] = x[0] * self.scale[i] + self.loc[i]
                x[1][...] = x[1] * self.scale[i]
            elif self.head == 'MixTruncatedNormal' or self.head == 'MixTruncatedNormal2':
                x[0][...] = x[0] * self.scale[i].unsqueeze(-1) + self.loc[i].unsqueeze(-1)
                x[1][...] = x[1] * self.scale[i].unsqueeze(-1)
            elif self.head == 'logNormal' or self.head == 'CrossEntropy':
                pass # no need to denormalize
            else:
                raise ValueError('head not supported, should be "RMSE", "NB", "logNormal", "Normal", "TruncatedNormal", '
                                 '"CrossEntropy", "MixTruncatedNormal", or "MixTruncatedNormal2"')
        return x


class RmseHead(nn.Module):
    def __init__(self, d_model, output_len, dropout=0, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, output_len)

    def forward(self, x):
        x = self.linear(self.dropout(x))
        return (x,)


class NormalHead(nn.Module):
    """Produces parameters for a normal distribution, log-Normal distribution, or truncated normal distribution."""
    def __init__(self, d_model, output_len, dropout=0, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mu = nn.Linear(d_model, output_len)
        self.sigma = nn.Linear(d_model, output_len)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: [bs x num_patch x d_model]
        mu = self.mu(self.dropout(x))
        sigma = self.softplus(self.sigma(self.dropout(x)))
        return mu, sigma

class MixNormalHead(nn.Module):
    def __init__(self, d_model, output_len, dropout=0, n_mixture=2, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mu_list = nn.ModuleList([nn.Linear(d_model, output_len) for _ in range(n_mixture)])
        self.sigma_list = nn.ModuleList([nn.Linear(d_model, output_len) for _ in range(n_mixture)])
        self.k = nn.Linear(d_model, n_mixture*output_len)
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.n_mixture = n_mixture
        self.output_len = output_len

    def forward(self, x):
        # x: [bs x num_patch x d_model]
        mu = [mu(self.dropout(x)) for mu in self.mu_list]
        sigma = [self.softplus(sigma(self.dropout(x))) for sigma in self.sigma_list]
        k = self.k(self.dropout(x))
        k = k.reshape(k.shape[0], k.shape[1], self.output_len, self.n_mixture)
        k = self.softmax(k)
        mu = torch.stack(mu, dim=-1)  # [bs x num_patch x output_len x n]
        sigma = torch.stack(sigma, dim=-1)  # [bs x num_patch x output_len x n]

        return mu, sigma, k


class MixNormalHead2(nn.Module):
    def __init__(self, d_model, output_len, dropout=0, n_mixture=2, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mu_list = nn.ModuleList([nn.Linear(d_model+i*3, 1) for i in range(n_mixture)])
        self.sigma_list = nn.ModuleList([nn.Linear(d_model+i*3, 1) for i in range(n_mixture)])
        self.k = nn.ModuleList([nn.Linear(d_model+i*3, 1) for i in range(n_mixture-1)])
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.n_mixture = n_mixture
        self.output_len = output_len

    def forward(self, x):
        # x: [bs x num_patch x d_model]
        remaining_k = torch.ones(x.shape[0], x.shape[1], 1, dtype=x.dtype).to(x.device)
        for i in range(self.n_mixture):
            current_mu = self.mu_list[i](self.dropout(x)) # [bs x num_patch x 1]
            current_sigma = self.softplus(self.sigma_list[i](self.dropout(x)))  # [bs x num_patch x 1]

            if i < self.n_mixture - 1:
                current_k = remaining_k * self.sigmoid(self.k[i](self.dropout(x)))  # [bs x num_patch x 1]
                remaining_k = remaining_k - current_k
            else:
                current_k = remaining_k

            if i == 0:
                mu_list = current_mu
                sigma_list = current_sigma
                k_list = current_k
            else:
                mu_list = torch.cat((mu_list, current_mu), dim=-1)
                sigma_list = torch.cat((sigma_list, current_sigma), dim=-1)
                k_list = torch.cat((k_list, current_k), dim=-1)

            if i < self.n_mixture - 1:
                x = torch.cat((x, current_mu, current_sigma, current_k), dim=-1)

        mu_list = mu_list.unsqueeze(-2)
        sigma_list = sigma_list.unsqueeze(-2)
        k_list = k_list.unsqueeze(-2)

        return mu_list, sigma_list, k_list


class HeadNegativeBinomial(nn.Module):
    def __init__(self, d_model, output_len, dropout=0, **kwargs):
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


class CrossEntropyHead(nn.Module):
    def __init__(self, d_model, output_len, dropout=0, num_bins=100, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, output_len*num_bins)
        self.num_bins = num_bins

    def forward(self, x):
        """x: [bs x num_patch x d_model]
        return: [bs x num_patch x output_len x num_bins]
        """
        x = self.linear(self.dropout(x))
        return (x.reshape(x.shape[0], x.shape[1], -1, self.num_bins),)


#%% Negative log likelihood
def nll_NB(n_success, p_success, y):
    """Compute the negative log likelihood of Negative Binomial distribution, element-wise."""
    nll = (- torch.lgamma(n_success + y) + torch.lgamma(n_success) +
           torch.lgamma(y + 1) - n_success * torch.log(p_success) -
           y * torch.log(1 - p_success))
    return nll

def nll_LogNormal(mu, sigma, y):
    """Compute the negative log likelihood of LogNormal distribution, element-wise."""
    nnl = torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5 * ((torch.log(y) - mu) / sigma) ** 2 + torch.log(y)
    return nnl

NORMAL_SCALE = np.log(2 * np.pi)*0.5
SQRT2 = np.sqrt(2)
def nll_truncated_normal(mu, sigma, y, a, b):
    """Compute the negative log likelihood of zero-truncated Normal distribution, element-wise.
    Here mu, sigma, y, a, b are all tensors. a, b are the lower and upper bounds of the truncated normal distribution.
    """
    alpha = (0 - mu) / sigma
    Z = 1 - 0.5 * (1 + torch.erf(alpha / SQRT2))
    Z = Z.clamp(min=1e-7)  # For numerical stability
    nnl = torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + NORMAL_SCALE + torch.log(Z)
    return nnl

def nll_normal(mu, sigma, y):
    """Compute the negative log likelihood of Normal distribution, element-wise."""
    nnl = torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + NORMAL_SCALE
    return nnl

def MixTruncatedNormal_loss(mu, sigma, k, y):
    """Compute the negative log likelihood of mixture of zero-truncated Normal distribution, element-wise.
    mu: [bs x num_patch x output_len x k]
    k: the mixing proportion of each component, sum(k) = 1
    """
    alpha = (0 - mu) / sigma
    Z = 1 - 0.5 * (1 + torch.erf(alpha / SQRT2))
    Z = Z.clamp(min=1e-7)  # For numerical stability
    l = torch.exp(-0.5 * ((y.unsqueeze(-1) - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi) * Z)  # [bs x num_patch x output_len x k]
    l = torch.sum(l * k, dim=-1).clamp(min=1e-10)  # [bs x num_patch x output_len]
    l = torch.mean(-torch.log(l))
    return l


def nb_loss(n_success, p_success, y, **kwargs):
    return torch.mean(nll_NB(n_success, p_success, y))

def logNormal_loss(mu, sigma, y, **kwargs):
    return torch.mean(nll_LogNormal(mu, sigma, y))

def rmse_loss(y_hat, y, **kwargs):
    error = y - y_hat
    loss = torch.sqrt(torch.mean(error ** 2))
    return loss


class cross_entropy_loss():
    def __init__(self, weights=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        else:
            self.weights = None
        self.loss = nn.CrossEntropyLoss(weight=self.weights)
    def __call__(self, y_hat, y):
        # y_hat: [bs x num_patch x output_len x num_bins]
        y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        y = y.reshape(-1)
        return self.loss(y_hat, y)


class multinomial_CRPS_loss():
    def __init__(self, num_bins):
        self.num_bins = num_bins
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bins_num = torch.arange(num_bins).reshape(1, 1, 1, num_bins).to(device)
    def __call__(self, y_hat, y):
        # y_hat: [bs x num_patch x output_len x num_bins], unnormailzed
        # y: [bs x num_patch x output_len], the real label
        y_hat = nn.Softmax(dim=-1)(y_hat)
        cum_y_hat = torch.cumsum(y_hat, dim=-1)  # cdf
        y = y.unsqueeze(-1)  # [bs x num_patch x output_len x 1]
        stepped = (self.bins_num >= y).to(torch.float32)  # [bs x num_patch x output_len x num_bins]
        crps = torch.sum((cum_y_hat - stepped) ** 2, dim=-1)
        return torch.mean(crps)


def truncated_normal_loss(mu, sigma, y, a=0, b=20000):
    return torch.mean(nll_truncated_normal(mu, sigma, y, a, b))

def normal_loss(mu, sigma, y, **kwargs):
    return torch.mean(nll_normal(mu, sigma, y))

#%% Sampling functions
class SampleNB:
    def __init__(self, **kwargs):
        pass
    def __call__(self, n_success, p_success):
        total_count = n_success
        probs = 1 - p_success
        return torch.distributions.NegativeBinomial(total_count, probs).sample((1,)).squeeze(0)

class SampleLogNormal:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu, sigma):
        dist = torch.distributions.LogNormal(mu, sigma)
        return dist.sample((1,)).squeeze(0)

class SampleRMSE:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu):
        return mu


class SampleCrossEntropy:
    def __init__(self, bin_edges, top_p=0.9, return_type='number', **kwargs):
        self.p = top_p  # nucleus sampling threshold
        self.return_type = return_type
        self.bin_edges = bin_edges
    def __call__(self, y_hat, fast=True):
        # y_hat: [bs x num_patch x output_len x num_bins]
        # Nucleus Sampling (Top-p sampling), include the first class that has cumulative probability > p
        # out: [bs x num_patch x output_len]
        y_hat = nn.Softmax(dim=-1)(y_hat)
        out = torch.zeros(y_hat.shape[:-1], dtype=torch.long, device=y_hat.device)
        if not fast:
            for i in range(y_hat.shape[0]):
                for j in range(y_hat.shape[1]):
                    for k in range(y_hat.shape[2]):
                        idx = torch.argsort(y_hat[i, j, k], descending=True)
                        cumsum = torch.cumsum(y_hat[i, j, k][idx], dim=-1)
                        cumsum = cumsum - y_hat[i, j, k][idx]  # exclude the current value
                        idx = idx[cumsum < self.p]
                        out[i, j, k] = idx[torch.multinomial(y_hat[i, j, k][idx], 1, replacement=True)]  # select one from the remaining classes by proportion
                        # out[i, j, k] = idx[torch.randint(0, len(idx), (1,))]  # select one from the remaining classes randomly
        if fast:
            out = torch.multinomial(y_hat.view(-1, y_hat.shape[-1]), 1, replacement=True)
            out = out.view(y_hat.shape[:-1])
        if self.return_type == 'bins':
            return out
        elif self.return_type == 'number':
            return bin2num(out, self.bin_edges)


class SampleTruncatedNormal:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu, sigma):
        out = torch.zeros(mu.shape, dtype=torch.float32, device=mu.device)
        idx = out == 0
        n = 0
        while (idx.sum()>0) & (n < 5):
            out[idx] = torch.distributions.Normal(mu[idx], sigma[idx]).sample((1,))
            idx = (out < 0)
            n += 1
        out = torch.clamp(out, 0)
        return out

class SampleMixTruncatedNormal:
    def __init__(self, **kwargs):
        self.sample = SampleTruncatedNormal()
    def __call__(self, mu, sigma, k):
        # mu: [bs x num_patch x output_len x k]
        # draw a random component from the mixture `k`
        idx = torch.distributions.Categorical(k).sample().unsqueeze(-1)
        mu = mu.gather(-1, idx).squeeze(-1)
        sigma = sigma.gather(-1, idx).squeeze(-1)
        out = self.sample(mu, sigma)
        return out

class SampleNormal:
    def __init__(self, **kwargs):
        pass
    def __call__(mu, sigma):
        dist = torch.distributions.Normal(mu, sigma)
        return dist.sample((1,)).squeeze(0)


#%% Mean function
class MeanRMSE:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu):
        return mu

class MeanNB:
    def __init__(self, **kwargs):
        pass
    def __call__(self, n_success, p_success):
        return n_success * (1 - p_success) / p_success

class MeanlogNormal:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu, sigma):
        return torch.exp(mu + sigma ** 2 / 2)

class MeanCrossEntropy:
    def __init__(self, bin2num_map, bin_edges, return_type='number', **kwargs):
        self.bin2num_map = bin2num_map  # [num_bins]
        self.bin_edges = bin_edges
        self.return_type = return_type
    def __call__(self, y_hat):
        # y_hat: [bs x num_patch x output_len x num_bins]
        # out: [bs x num_patch x output_len]
        y_hat = nn.Softmax(dim=-1)(y_hat)
        y_hat = y_hat * self.bin2num_map
        y_hat = torch.sum(y_hat, dim=-1)
        # Whether to return the bin index or the number depends on the input type of the next step forecast
        if self.return_type == 'number':
            return y_hat
        elif self.return_type == 'bins':  # return the bin index
            return num2bin(y_hat, self.bin_edges)

class MeanTruncatedNormal:
    def __init__(self, **kwargs):
        self.eps = torch.finfo(torch.float32).eps
    def __call__(self, mu, sigma):
        alpha = (0 - mu) / sigma
        Z = 1 - 0.5 * (1 + torch.erf(alpha / SQRT2))
        Z = Z.clamp(min=self.eps)  # For numerical stability
        out = mu + sigma /Z * (torch.exp(-0.5 * alpha ** 2) - 0) / np.sqrt(2 * np.pi)
        return out

class MeanMixTruncatedNormal:
    def __init__(self, **kwargs):
        self.mean = MeanTruncatedNormal()
    def __call__(self, mu, sigma, k):
        mu = self.mean(mu, sigma) # [bs x num_patch x output_len x k]
        out = torch.sum(mu * k, dim=-1) # [bs x num_patch x output_len]
        return out

class MeanNormal:
    def __init__(self, **kwargs):
        pass
    def __call__(self, mu, sigma):
        return mu

#%% aggregations
head_dic = {'NB': HeadNegativeBinomial, 'logNormal': NormalHead, 'RMSE': RmseHead, 'CrossEntropy': CrossEntropyHead,
            'TruncatedNormal': NormalHead, 'Normal': NormalHead, 'MixTruncatedNormal': MixNormalHead,
            'MixTruncatedNormal2': MixNormalHead2}
mean_dic = {'NB': MeanNB, 'logNormal': MeanlogNormal, 'RMSE': MeanRMSE, 'CrossEntropy': MeanCrossEntropy,
            'TruncatedNormal': MeanTruncatedNormal, 'Normal': MeanNormal,
            'MixTruncatedNormal': MeanMixTruncatedNormal, 'MixTruncatedNormal2': MeanMixTruncatedNormal}
sample_dic = {'NB': SampleNB, 'logNormal': SampleLogNormal, 'RMSE': SampleRMSE, 'CrossEntropy': SampleCrossEntropy,
              'TruncatedNormal': SampleTruncatedNormal, 'Normal': SampleNormal,
              'MixTruncatedNormal': SampleMixTruncatedNormal, 'MixTruncatedNormal2': SampleMixTruncatedNormal}

def get_loss(head, a=0, b=20000, weights=None, train_method=None, num_bins=None, **kwargs):
    if head == 'RMSE':
        return rmse_loss
    elif head == 'NB':
        return nb_loss
    elif head == 'logNormal':
        return logNormal_loss
    elif head == 'CrossEntropy' and train_method != 'CRPS':
        loss = cross_entropy_loss(weights=weights)
        return loss
    elif head == 'CrossEntropy' and train_method == 'CRPS':
        return multinomial_CRPS_loss(num_bins=num_bins)
    elif head == 'TruncatedNormal':
        return lambda mu, sigma, y: truncated_normal_loss(mu, sigma, y, a=a, b=b)
    elif head == 'Normal':
        return normal_loss
    elif head == 'MixTruncatedNormal':
        return MixTruncatedNormal_loss
    elif head == 'MixTruncatedNormal2':
        return MixTruncatedNormal_loss
    else:
        raise ValueError('loss not supported')

def train_model(args, train_loader, val_loader, test_loader, model, device='cuda:0'):
    trian_start_time = time.time()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = get_loss(args.head_type, a=0, **vars(args))

    run = wandb.init(project='ABTransformer', config=dict(args._get_kwargs()), reinit=True, mode=args.mode)
    now = (f'{datetime.datetime.now().month:02d}_{datetime.datetime.now().day:02d}_{datetime.datetime.now().hour:02d}'
           f'_{datetime.datetime.now().minute:02d}_{datetime.datetime.now().second:02d}')
    wandb.run.name = now + f'_{args.dset}_{args.model}'
    args.name = run.name

    # learning rate scheduler
    step_loss = []
    epoch_loss = []
    lrs = []
    patience = args.patience
    best_val_loss = np.inf
    epochs_no_improve = 0
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.n_epochs, pct_start=1/args.n_epochs,
                                                    div_factor=args.div_factor, final_div_factor=args.final_div_factor,
                                                    anneal_strategy=args.anneal_strategy)

    for epoch in range(args.n_epochs):
        model.train()
        for i, (inputs, target) in enumerate(train_loader):
            if args.head_type == 'logNormal':  # logNormal head has a problem with 0 values
                target = target+0.01
            inputs = [input.to(device) for input in inputs]
            target = target.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(*output, target)
            step_loss.append(loss.item())
            lrs.append(scheduler.get_last_lr()[0])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 300 == 0:
                print(f'Epoch [{epoch}/{args.n_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}'
                      f'\t total time: {time.time() - trian_start_time:.2f}')

        epoch_loss.append(np.mean(step_loss[-len(train_loader):]))

        # Calculate validation loss
        model.eval()
        val_loss = []
        for i, (inputs, target) in enumerate(val_loader):
            if args.head_type == 'logNormal':  # logNormal head has a problem with 0 values
                target = target+0.01
            inputs = [input.to(device) for input in inputs]
            target = target.to(device)
            output = model(inputs)
            loss = criterion(*output, target)
            val_loss.append(loss.item())
        print(f'Epoch [{epoch}/{args.n_epochs}], Val Loss: {np.mean(val_loss):.4f} \t Train Loss: {epoch_loss[-1]:.4f} '
              f'\t total time: {time.time() - trian_start_time:.2f}')
        best_val_loss = min(best_val_loss, np.mean(val_loss))
        wandb.log({'train_loss': epoch_loss[-1], 'val_loss': np.mean(val_loss), 'epoch': epoch })

        # Save the current best model
        if np.mean(val_loss) == best_val_loss:
            torch.save(model.state_dict(), f'log\\{args.name}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping!')
            break
        if time.time() - trian_start_time > args.max_run_time*3600:
            print(f'Time limit {args.max_run_time} hours reached! Stopping training.')
            break

    # Load the best model
    model.load_state_dict(torch.load(f'log\\{args.name}.pth'))

    model.eval()
    test_loss = []
    test_loader.dataset.test_mode = True
    with torch.no_grad():
        for x, y, abnormal in test_loader:
            x = [xx.to(device) for xx in x]
            y = y.to(device)
            y_hat = model.forecast(x)
            loss = rmse_loss(y_hat, y[:, -args.num_target_patch:, :])
            test_loss.append(loss.item()**2)
    print(f'Test Loss: {np.sqrt(np.mean(test_loss)):.4f}')
    wandb.log({'test_loss': np.sqrt(np.mean(test_loss))})

    wandb.finish()
    return model
