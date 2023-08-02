import torch
import torch.nn as nn
import numpy as np

class Standardization(nn.Module):
    def __init__(self, loc, scale):
        super(Standardization, self).__init__()
        self.loc = loc.reshape([-1, 1])
        self.scale = scale.reshape([-1, 1])

    def forward(self, x, i, mode: str):
        """
        x: (bs, context_len)
        i: index of the patch, (bs,)
        """
        if mode == 'norm':
            x = (x - self.loc[i]) / self.scale[i]
        elif mode == 'denorm':
            x = x * self.scale[i] + self.loc[i]
        else:
            raise NotImplementedError
        return x


class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, x_loc, x_scale, args):
        super(NLinear, self).__init__()
        self.context_len = args.context_len * 2  # Alighting + Boarding
        self.target_len = args.target_len

        self.n_stations = x_loc.size(0)
        self.Linear = nn.ModuleList()
        self.standardization = Standardization(x_loc, x_scale)
        for i in range(self.n_stations):
            self.Linear.append(nn.Linear(self.context_len,self.target_len))
        self.softplus = nn.Softplus()

    def forward(self, x):

        # x: (Batch x Input length, Batch of station_num)
        z = x[0]
        stations = x[1].ravel().long()
        z = self.standardization(z, stations, 'norm')

        output = torch.zeros([z.size(0),self.target_len],dtype=z.dtype).to(z.device)
        for i in range(stations.size(0)):
            output[i,:] = self.Linear[stations[i]](z[i,:])
        output = self.standardization(output, stations, 'denorm')
        output = self.softplus(output)

        return output
