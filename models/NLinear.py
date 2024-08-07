import torch
import torch.nn as nn
import numpy as np
from .basics import Standardization

# class NLinear(nn.Module):
#     """
#     Normalization-Linear (NLinear) model with weekday and time_in_day embeddings,
#     The no-embedding version of NLinear is from https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
#     """
#     def __init__(self, num_embeds, x_loc, x_scale, input_len, target_len,
#                  d_model, forecast_target='both', dropout=0.05, **kwargs):
#         super(NLinear, self).__init__()
#         self.input_len = input_len  # Alighting + Boarding
#         self.target_len = target_len
#         self.x_loc = x_loc
#         self.x_scale = x_scale
#         self.d_model = d_model
#
#         if forecast_target == 'both':
#             self.factor = 2
#         elif forecast_target in {'inflow', 'outflow'}:
#             self.factor = 1
#
#         self.n_stations = x_loc.size(0)
#         self.dropout = nn.Dropout(dropout)
#         self.in_projection = nn.Linear(self.input_len*self.factor, self.d_model)
#         self.station_eb = nn.Embedding(num_embeds[0], self.d_model)
#         self.week_eb = nn.Embedding(num_embeds[1], self.d_model)
#         self.time_eb = nn.Embedding(num_embeds[2], self.d_model)
#         self.out_projection = nn.Linear(self.d_model, self.target_len*self.factor)
#         self.standardization = Standardization(x_loc, x_scale)
#         self.softplus = nn.Softplus()
#
#     def forward(self, x):
#         # x: (bs x Input length, (patch_len or patch_len*2))
#         z = x[0]
#         stations = x[1][:, 0].long()
#         z = self.standardization(z, stations, 'norm')
#         z = z.view(-1, self.input_len*self.factor)
#
#         z = self.dropout(self.in_projection(z))
#         z = z + self.dropout(self.station_eb(stations))
#         z = z + self.dropout(self.week_eb(x[2][:, 0].long()))
#         z = z + self.dropout(self.time_eb(x[3][:, 0].long()))
#         output = self.dropout(self.out_projection(z))
#
#         output = output.view(-1, self.target_len, self.factor)
#         output = self.standardization((output,), stations, 'denorm')
#
#         return output
#
#     def forecast(self, x):
#         return self.forward(x)[0]


# class NLinear(nn.Module):
#     """
#     Normalization-Linear (NLinear) model with weekday and time_in_day embeddings,
#     The no-embedding version of NLinear is from https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
#     """
#     def __init__(self, num_embeds, x_loc, x_scale, input_len, target_len,
#                  d_model, forecast_target='both', dropout=0.05, **kwargs):
#         super(NLinear, self).__init__()
#         self.input_len = input_len  # Alighting + Boarding
#         self.target_len = target_len
#         self.x_loc = x_loc
#         self.x_scale = x_scale
#         self.d_model = d_model
#
#         if forecast_target == 'both':
#             self.factor = 2
#         elif forecast_target in {'inflow', 'outflow'}:
#             self.factor = 1
#
#         self.n_stations = x_loc.size(0)
#         self.dropout = nn.Dropout(dropout)
#         self.in_projection = nn.ModuleList(nn.Linear(self.input_len*self.factor, self.d_model) for _ in range(self.n_stations))
#         self.week_eb = nn.ModuleList(nn.Embedding(num_embeds[1], self.d_model) for _ in range(self.n_stations))
#         self.time_eb = nn.ModuleList(nn.Embedding(num_embeds[2], self.d_model) for _ in range(self.n_stations))
#         self.out_projection = nn.ModuleList(nn.Linear(self.d_model, self.target_len*self.factor) for _ in range(self.n_stations))
#         self.standardization = Standardization(x_loc, x_scale)
#         self.softplus = nn.Softplus()
#
#     def forward(self, x):
#         # x: (bs x Input length, (patch_len or patch_len*2))
#         z = x[0]
#         stations = x[1][:, 0].long()
#         z = self.standardization(z, stations, 'norm')
#         z = z.view(-1, self.input_len*self.factor)
#
#         output = torch.zeros([z.size(0), self.target_len*self.factor], dtype=z.dtype).to(z.device)
#         for i in range(stations.size(0)):
#             y = self.dropout(self.in_projection[stations[i]](z[i,:]))
#             y += self.dropout(self.week_eb[stations[i]](x[2][i, 0].long()))
#             y += self.dropout(self.time_eb[stations[i]](x[3][i, 0].long()))
#             output[i, :] = self.dropout(self.out_projection[i](y))
#
#         output = output.view(-1, self.target_len, self.factor)
#         output = self.standardization((output,), stations, 'denorm')
#
#         return output
#
#     def forecast(self, x):
#         return self.forward(x)[0]


class NLinear(nn.Module):
    """
    Normalization-Linear (NLinear) model with weekday and time_in_day embeddings,
    The no-embedding version of NLinear is from https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
    """
    def __init__(self, num_embeds, x_loc, x_scale, input_len, target_len,
                 d_model, forecast_target='both', dropout=0.05, **kwargs):
        super(NLinear, self).__init__()
        self.input_len = input_len  # Alighting + Boarding
        self.target_len = target_len
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.d_model = d_model

        if forecast_target == 'both':
            self.factor = 2
        elif forecast_target in {'inflow', 'outflow'}:
            self.factor = 1

        self.n_stations = x_loc.size(0)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.ModuleList(nn.Linear(self.input_len*self.factor + self.d_model, self.target_len*self.factor) for _ in range(self.n_stations))
        self.week_eb = nn.ModuleList(nn.Embedding(num_embeds[1], self.d_model) for _ in range(self.n_stations))
        self.time_eb = nn.ModuleList(nn.Embedding(num_embeds[2], self.d_model) for _ in range(self.n_stations))
        self.standardization = Standardization(x_loc, x_scale)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (bs x Input length, (patch_len or patch_len*2))
        z = x[0]
        stations = x[1][:, 0].long()
        z = self.standardization(z, stations, 'norm')
        z = z.view(-1, self.input_len*self.factor)

        output = torch.zeros([z.size(0), self.target_len*self.factor], dtype=z.dtype).to(z.device)
        for i in range(stations.size(0)):
            d = self.dropout(self.week_eb[stations[i]](x[2][i, 0].long()))
            d += self.dropout(self.time_eb[stations[i]](x[3][i, 0].long()))
            y = torch.cat([z[i,:], d], dim=-1)
            output[i, :] = self.dropout(self.projection[stations[i]](y))

        output = output.view(-1, self.target_len, self.factor)
        output = self.standardization((output,), stations, 'denorm')

        return output

    def forecast(self, x):
        return self.forward(x)[0]