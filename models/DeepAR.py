import torch
import torch.nn as nn
from .basics import *


class DeepAR(nn.Module):
    """The deepAR only use the final hidden state of LSTM to predict the next value of a"""
    def __init__(self, num_embeds, d_model, n_layers, num_target_patch, x_loc=None, x_scale=None, patch_len=1,
                 dropout=0.05, head_dropout=0, input_type='number', head_type='RMSE', input_emb_size=8, num_bins=None,
                 bin_edges=None, top_p=0.9, n_mixture=2, **kwargs):
        '''
        A recurrent network that predicts the future values of a time-dependent variable based on past inputs.
        '''
        super(DeepAR, self).__init__()
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.patch_len = patch_len
        self.num_target_patch = num_target_patch
        self.input_type = input_type
        self.head_type = head_type
        self.num_bins = num_bins
        self.num_embeds = num_embeds
        self.d_model = d_model
        self.n_layers = n_layers

        # Standardization
        self.standardization = Standardization(self.x_loc, self.x_scale, input_type, head_type)

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if input_type == 'number':
            self.W_P = nn.Linear(patch_len * 2, d_model)
        elif input_type == 'bins':
            self.W_P = nn.Sequential(nn.Embedding(num_bins, input_emb_size), FlattenLastTwoDims(), nn.Linear(input_emb_size*2, d_model))

        # Flow_type, station, weekday, or time_in_day embedding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])

        # LSTM
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_model,
                            num_layers=n_layers,
                            bias=True,
                            batch_first=True,
                            dropout=dropout)

        # The output head part
        if self.head_type == 'CrossEntropy':
            self.bin2num_map = torch.nn.Parameter(torch.tensor([bin2num(x, bin_edges) for x in range(num_bins)], dtype=torch.float32).squeeze())
            self.bin_edges = torch.nn.Parameter(torch.from_numpy(bin_edges), requires_grad=False)
        else:
            self.bin2num_map = None
            self.bin_edges = None

        self.head = head_dic[head_type](d_model, patch_len * 2, head_dropout, num_bins=num_bins, n_mixture=n_mixture)
        self.mean = mean_dic[head_type](bin2num_map=self.bin2num_map, bin_edges=self.bin_edges, return_type=self.input_type)
        self.sample = sample_dic[head_type](bin_edges=self.bin_edges, top_p=top_p, return_type=self.input_type)

    def forward(self, x, state=None, method='param'):
        '''
        Args:
            x: [batch_size, seq_length, patch_len*2] + a list of features of shape [batch_size, seq_length]
            state: initial state of (hidden, cell), default None means (zeros)
            method: 'param','mean' or 'sample'
        Returns:
            z: ([batch_size, seq_length, patch_len*2])
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        # Initialization
        z = x[0]  # z: [bs x num_patch x patch_len*2]
        features = x[1:]
        stations = x[1][:, 0].long()

        # Input encoding
        z = self.standardization(z, stations, 'norm')
        z = self.W_P(z)  # z: [bs x num_patch x d_model]

        for i, feature_eb in enumerate(self.feature_eb):
            z += feature_eb(features[i])  # z: [bs x num_patch x d_model]

        # LSTM
        if state is None:
            z, (hidden, cell) = self.lstm(z)
        else:
            hidden, cell = state
            z, (hidden, cell) = self.lstm(z, (hidden, cell))  # z: [bs x num_patch x d_model]

        # Output decoding
        z = self.head(z)  # a tuple of [bs x num_fcst x patch_len]

        z = self.standardization(z, stations, 'denorm')

        if method=='param':
            return z
        elif method=='mean':
            return self.mean(*z), (hidden, cell)
        elif method=='sample':
            return self.sample(*z), (hidden, cell)
        else:
            raise ValueError("method should be 'param', 'mean', or 'sample'.")

    def forecast(self, x, method='mean'):
        """Autoregressive forecasting in the test phase
        method: 'mean' or 'sample'
        """
        # Here I assume that deepAR are always patch_len=1
        n_target = self.num_target_patch  # number of patches to predict
        n_input = x[0].shape[1] - n_target + 1
        features = x[1:]

        result = []
        self.eval()
        with torch.no_grad():
            xx = [x[0][:, :n_input, :]] + [feature[:, :n_input] for feature in features]
            y_new, state = self(xx, method=method)
            for i in range(n_target-1):
                y_new = y_new[:, [-1], :]
                result.append(y_new)
                xx = [y_new] + [feature[:, [n_input + i]] for feature in features]
                y_new, state = self(xx, state, method=method)
            result.append(y_new)
            result = torch.cat(result, dim=1)

        if self.head_type == 'CrossEntropy' and self.input_type == 'bins':
            result = self.bin2num_map[result]

        return result

    def forecast_samples(self, x, n=100):
        """Autoregressive forecasting in the test phase, draw n samples
        """
        result = []
        with torch.no_grad():
            for i in range(n):
                result.append(self.forecast(x=x, method='sample'))
        return torch.stack(result, dim=0)

