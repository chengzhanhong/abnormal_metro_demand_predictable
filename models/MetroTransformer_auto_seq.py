__all__ = ['MetroTransformer_v']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from .pos_encoding import *
from .basics import *
from .attention import *
from .revin import *


# Cell
class MetroTransformer_auto_seq(nn.Module):
    """
    Output dimension:
         [bs x target_len] for prediction
         [bs x num_patch x patch_len] for pretrain
    """

    def __init__(self, patch_len: int, num_patch: int, num_target_patch, num_embeds: tuple = (2, 159, 7, 217),
                 n_layers: int = 3, d_model=128, n_heads=16, d_ff: int = 256,
                 norm: str = 'LayerNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0,
                 attn_mask=None, x_loc=None, x_scale=None, **kwargs):
        """
        Parameters:
            num_embeds: tuple of number of embeddings for flow_type, station, weekday, or time_in_day
            d_ff: dimension of the inner feed-forward layer
            pe: type of positional encoding initialization, "zeros", "sincos", or None
            learn_pe: whether to learn positional encoding
            revin: whether to use reversible instance normalization
            ABflow: whether to use AB flow model
        """
        super().__init__()

        self.num_patch = num_patch
        self.patch_len = patch_len
        self.num_target_patch = num_target_patch
        self.attn_mask = attn_mask
        self.x_loc = x_loc
        self.x_scale = x_scale

        # Backbone
        self.backbone = MetroEncoder(num_patch=num_patch, patch_len=patch_len, num_target_patch=num_target_patch,
                                     num_embeds=num_embeds, n_layers=n_layers, d_model=d_model,
                                     n_heads=n_heads, d_ff=d_ff, attn_dropout=attn_dropout,
                                     dropout=dropout, act=act, pre_norm=pre_norm, store_attn=store_attn,
                                     norm=norm, pe=pe, learn_pe=learn_pe)

        self.head = ForecastHead(d_model, patch_len, head_dropout)
        # Standardization
        self.standardization = Standardization(self.x_loc, self.x_scale)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        x: tuple of flow tensor [bs x num_patch x patch_len] and feature tensors of [bs x num_patch x 1].
        """
        z = x[0]
        features = x[1:]
        stations = features[1][:, 0].long()

        z = self.standardization(z, stations, 'norm')
        z = self.backbone(z, features, self.attn_mask)  # z: [bs x (num_patch+num_target_patch) x d_model]
        z = self.head(z, self.num_target_patch+self.num_patch-1)  # z: [bs x num_fcst x patch_len]
        z = self.standardization(z, stations, 'denorm')  # z: [bs x num_fcst x patch_len]
        z = self.softplus(z)
        return z

    def forecast(self, x):
        """
        Auto-regressive forecasting.
        x: tuple of flow tensor [bs x num_patch x patch_len] and feature tensors of [bs x num_patch x 1].
        """
        self.eval()
        N = self.num_target_patch  # number of patches to predict
        y = x[0].clone()  # The input and also the prediction
        features = x[1:]
        stations = features[1][:, 0].long()

        for i in range(N):
            z = self.standardization(y, stations, 'norm')
            z = self.backbone(z, features, self.attn_mask)  # z: [bs x (num_patch+num_target_patch) x d_model]
            z = self.head(z, N)  # z: [bs x num x patch_len]
            z = self.standardization(z, stations, 'denorm')  # z: [bs x num_fcst x patch_len]
            z = self.softplus(z)
            if i < N - 1:
                y[:, -N+i+1, :] = z[:, -N+i, :]

        return z[:, -N:, :]


class MetroEncoder(nn.Module):
    def __init__(self, num_patch, num_target_patch, patch_len, num_embeds, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 pre_norm=False, pe='zeros', learn_pe=True):
        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_target_patch+num_patch*2-1, d_model)

        # Flow_type, station, weekday, or time_in_day encoding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, n_layers=n_layers, store_attn=store_attn)

    def forward(self, z, features, attn_mask=None) -> Tensor:
        """
        z: tensor [bs x num_patch x patch_len]
        features: tuple of tensor [bs x num_patch], representing flow_type, station, weekday, or time_in_day
        """
        z = self.W_P(z)  # z: [bs x num_patch x d_model]
        z = z + self.W_pos

        # feature encoding
        for i, feature_eb in enumerate(self.feature_eb):
            z += feature_eb(features[i])  # z: [bs x num_patch x d_model]

        z = self.dropout(z)
        # Encoder
        z = self.encoder(z, attn_mask)  # z: [bs x num_patch x d_model]
        return z


# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     activation=activation,
                                                     pre_norm=pre_norm, store_attn=store_attn) for i in
                                     range(n_layers)])

    def forward(self, src: Tensor, attn_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        for mod in self.layers:
            output = mod(output, attn_mask)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0., dropout=0., bias=True,
                 activation="gelu", pre_norm=False):
        """pre_norm: if True, apply normalization before residual and multi-head attention."""
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        elif "layer" in norm.lower():
            self.norm_attn = nn.LayerNorm(d_model)
        else:
            raise NotImplementedError(f"Normalization {norm} not implemented")

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        elif "layer" in norm.lower():
            self.norm_ffn = nn.LayerNorm(d_model)
        else:
            raise NotImplementedError(f"Normalization {norm} not implemented")

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, attn_mask=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return src


class ForecastHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x, target_length):
        """
        x: tensor [bs x num_patch x d_model]
        output: tensor [bs x target_length x patch_len]
        """
        x = x[:, -target_length:, :]  # [bs x target_length x d_model]
        x = self.linear(self.dropout(x))  # [bs x target_length x patch_len]
        return x