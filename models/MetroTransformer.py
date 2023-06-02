__all__ = ['MetroTransformer']

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
class MetroTransformer(nn.Module):
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
                 attn_mask=None, x_loc=None, x_scale=None, station_rank=0, stride=None,
                 loss='rmse', **kwargs):
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

        num_patch = num_patch * 2
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.num_target_patch = num_target_patch
        self.attn_mask = attn_mask
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.loss = loss

        # Backbone
        self.backbone = MetroEncoder(num_patch=num_patch, patch_len=patch_len, num_target_patch=num_target_patch,
                                     num_embeds=num_embeds, n_layers=n_layers, d_model=d_model,
                                     n_heads=n_heads, d_ff=d_ff, attn_dropout=attn_dropout,
                                     dropout=dropout, act=act, pre_norm=pre_norm, store_attn=store_attn,
                                     norm=norm, pe=pe, learn_pe=learn_pe, station_rank=station_rank)

        # Head
        output_patch_len = patch_len - (patch_len - stride)
        output_dims = {'rmse': 1, 'mae': 1, 'gaussian_nll': 2, 'quantile1': 1, 'quantile3': 3, 'quantile5': 5}
        self.output_dim = output_dims[loss]
        self.head = ForecastHead(d_model, output_patch_len, head_dropout, self.output_dim)

        # Standardization
        self.standardization = Standardization(self.x_loc, self.x_scale)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        x: tuple of flow tensor [bs x num_patch x patch_len] and feature tensors of [bs x num_patch x 1].
        """
        z = x[0]
        features = x[1:-1]
        fcst_loc = x[-1]
        stations = features[1][:, 0].long()
        z = self.standardization(z, stations, 'norm')
        # todo: test post_zero
        # if self.post_zero:
        #     z[fcst_loc==1] = 0

        z = self.backbone(z, features, self.attn_mask)  # z: [bs x (num_patch+num_target_patch) x d_model]
        z = self.head(z, fcst_loc)  # z: [bs x num_fcst x patch_len x output_dim]

        if self.loss == 'gaussian_nll':
            z_clone = z[:, :, :, 0].clone()
            z_clone = self.standardization(z_clone, stations, 'denorm')
            z[:, :, :, 0] = self.softplus(z_clone)
            z_clone1 = z[:, :, :, 1].clone()
            z_clone1 = self.softplus(z_clone1)
            z[:, :, :, 1] = self.standardization(z_clone1, stations, 'descale')
        else:
            for i in range(self.output_dim):
                z[:, :, :, i] = self.standardization(z[:, :, :, i], stations,
                                                     'denorm')  # z: [bs x num_fcst x patch_len x *]
            z = self.softplus(z)
        return z


class Standardization(nn.Module):
    def __init__(self, loc, scale):
        super(Standardization, self).__init__()
        self.loc = loc.reshape([-1, 1, 1])
        self.scale = scale.reshape([-1, 1, 1])

    def forward(self, x, i, mode: str):
        """
        x: (bs, num_patch, patch_len)
        i: index of the patch, (bs,)
        """
        if mode == 'norm':
            x = (x - self.loc[i]) / self.scale[i]
        elif mode == 'denorm':
            x = x * self.scale[i] + self.loc[i]
        elif mode == 'descale':
            x = x * self.scale[i]
        else:
            raise NotImplementedError
        return x


class LowRankEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rank):
        super(LowRankEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank

        # Define the factor matrices
        self.W1 = nn.Parameter(torch.Tensor(num_embeddings, rank))
        self.W2 = nn.Parameter(torch.Tensor(rank, embedding_dim))
        self.intialize()

    def intialize(self):
        # Initialize the factor matrices
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, input):
        # Compute the low-rank weight matrix
        weight = torch.matmul(self.W1, self.W2)

        # Perform the embedding lookup
        embedded = torch.embedding(weight, input)

        return embedded


class MetroEncoder(nn.Module):
    def __init__(self, num_patch, num_target_patch, patch_len, num_embeds, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 pre_norm=False, pe='zeros', learn_pe=True, station_rank=0, **kwargs):

        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch + num_target_patch, d_model)

        # Flow_type, station, weekday, or time_in_day encoding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])
        if station_rank > 0:
            self.feature_eb[1] = LowRankEmbedding(num_embeds[1], d_model, station_rank)

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
    def __init__(self, d_model, patch_len, dropout, output_dim=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len * output_dim)
        self.patch_len = patch_len
        self.output_dim = output_dim

    def forward(self, x, fcst_loc):
        """
        x: tensor [bs x num_patch x d_model]
        output: tensor [bs x num_fcst x patch_len]
        """
        x = torch.gather(x, dim=1, index=fcst_loc.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [bs x num_fcst x d_model]
        x = self.linear(self.dropout(x))  # [bs x num_fcst x patch_len*output_dim]
        x = x.view(x.size(0), x.size(1), -1, self.output_dim)  # [bs x num_fcst x patch_len x output_dim]
        return x


class PatchTSTHead(nn.Module):
    """The head for the PatchTST model"""
    def __init__(self, d_model, patch_len, num_patch, num_target_patch, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.num_patch = num_patch
        self.num_target_patch = num_target_patch
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model*num_patch, patch_len*num_target_patch)
        self.flatten = nn.Flatten(start_dim=-2)

    def foward(self, x):
        """
        x: [bs x d_model*num_patch]
        output: [bs x num_target_patch x patch_len]
        """
        x = self.flatten(x)  # [bs x d_model*num_patch]
        x = self.linear(self.dropout(x))  # [bs x patch_len*num_target_patch]
        x = x.view(x.size(0), self.num_target_patch, self.patch_len)  # [bs x num_target_patch x patch_len]
        return x


