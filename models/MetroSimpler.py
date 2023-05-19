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
class MetroSimpler(nn.Module):
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

        num_patch = num_patch * 2
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

        # Head
        self.head = nn.Linear(d_model, patch_len*num_target_patch)

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
        # Debug
        z = self.backbone(z, features, self.attn_mask)  # z: [bs x (num_patch+num_target_patch) x d_model]

        z = self.head(z)  # z: [bs x num_fcst x patch_len]
        z = z.reshape(-1, self.num_target_patch, self.patch_len)
        z = self.standardization(z, stations, 'denorm')  # z: [bs x num_fcst x patch_len]
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
        else:
            raise NotImplementedError
        return x


class MetroEncoder(nn.Module):
    def __init__(self, num_patch, num_target_patch, patch_len, num_embeds, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 pre_norm=False, pe='zeros', learn_pe=True):
        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len * 2, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, 48, d_model)

        # Flow_type, station, weekday, or time_in_day encoding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # # Encoder
        # self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
        #                           pre_norm=pre_norm, activation=act, n_layers=n_layers, store_attn=store_attn)

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
        # Pooler
        z = z.permute(0, 2, 1)
        z = F.max_pool1d(z, kernel_size=z.size(-1), stride=3).reshape(-1, self.d_model)  # z: [bs x d_model]
        return z

