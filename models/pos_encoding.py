__all__ = ['PositionalEncoding', 'SinCosPosEncoding', 'positional_encoding']

# Cell

import torch
from torch import nn
import math


# Cell
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None or pe == 'none' or pe == 'rotary' or pe == 'rotary_half':
        W_pos = torch.zeros((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        learn_pe = False

    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)

    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
        learn_pe = False
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: \
        'zeros', 'sincos', 'rotary', 'rotary_half', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)
