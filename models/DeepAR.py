import torch
import torch.nn as nn
from .basics import *


class DeepAR(nn.Module):
    """The deepAR only use the final hidden state of LSTM to predict the next value of a"""
    def __init__(self, num_embeds, d_model, n_layers, num_target_patch, x_loc=None, x_scale=None, patch_len=1,
                 dropout=0.05, head_dropout=0, head='RMSE',**kwargs):
        '''
        A recurrent network that predicts the future values of a time-dependent variable based on past inputs.
        '''
        super(DeepAR, self).__init__()
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.patch_len = patch_len
        self.num_target_patch = num_target_patch

        self.standardization = Standardization(self.x_loc, self.x_scale)
        self.d_model = d_model
        self.n_layers = n_layers
        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len * 2, d_model)
        self.num_embeds = num_embeds
        # Flow_type, station, weekday, or time_in_day encoding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])

        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_model,
                            num_layers=n_layers,
                            bias=True,
                            batch_first=True,
                            dropout=dropout)
        self.head = head_dic['head'](d_model, patch_len * 2, head_dropout)
        self.softplus = nn.Softplus()

    def forward(self, x, state=None):
        '''
        Args:
            x: [batch_size, seq_length, patch_len*2] + a list of features of shape [batch_size, seq_length]
            state: initial state of (hidden, cell), default None means (zeros)
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
        z = self.head(z)  # z: [bs x num_fcst x patch_len]
        z = self.standardization(z, stations, 'denorm')  # z: [bs x num_fcst x patch_len]
        z = self.softplus(z)
        return z, (hidden, cell)

    def forecast(self, x):
        """Autoregressive forecasting in the test phase"""
        # Here I assume that deepAR are always patch_len=1
        n_target = self.num_target_patch  # number of patches to predict
        n_input = x[0].shape[1] - n_target + 1
        features = x[1:]

        result = []
        self.eval()
        with torch.no_grad():
            xx = [x[0][:, :n_input, :]] + [feature[:, :n_input] for feature in features]
            y_new, state = self(xx)
            for i in range(n_target-1):
                y_new = y_new[:, [-1], :]
                result.append(y_new)
                xx = [y_new] + [feature[:, [n_input + i]] for feature in features]
                y_new, state = self(xx, state)
            result.append(y_new)
            result = torch.cat(result, dim=1)

        return result
