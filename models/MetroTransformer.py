
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
    def __init__(self, target_len:int, patch_len:int, num_patch:int, num_embeds:tuple=(2,159,7,217),
                 n_layers:int=3, d_model=128, n_heads=16, d_ff:int=256,
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", output_dim=1, revin=False, ABflow=True, **kwargs):
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
        if ABflow:
            num_patch = num_patch*2
        self.num_patch = num_patch
        assert head_type in ['pretrain', 'prediction'], 'head type should be either pretrain or prediction'
        # Backbone
        self.backbone = MetroEncoder(num_patch=num_patch, patch_len=patch_len,num_embeds=num_embeds,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                pre_norm=pre_norm, store_attn=store_attn, norm=norm,
                                pe=pe, learn_pe=learn_pe)

        # Head
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout)
        elif head_type == "prediction":
            self.head = PredictionHead(d_model, num_patch, target_len, head_dropout=head_dropout, out_dim=output_dim)

        # Reversible Instance Normalization
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(1)


    def forward(self, x):
        """
        x: tuple of flow tensor [bs x num_patch x patch_len] and feature tensors of [bs x num_patch x 1].
        """
        z = x[0]
        features = x[1:]
        if self.revin:
            z = self.revin_layer(z,'norm')

        z = self.backbone(z, features)                                 # z: [bs x d_model x num_patch]
        z = self.head(z)

        if self.revin:
            z = self.revin_layer(z, 'denorm')

        # z: [bs x target_len] for prediction
        #    [bs x num_patch x patch_len] for pretrain
        return z.squeeze()


class MetroEncoder(nn.Module):
    def __init__(self, num_patch, patch_len, num_embeds, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 pre_norm=False, pe='zeros', learn_pe=True):

        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Flow_type, station, weekday, or time_in_day encoding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, n_layers=n_layers, store_attn=store_attn)

    def forward(self, z, features) -> Tensor:
        """
        z: tensor [bs x num_patch x patch_len]
        features: tuple of tensor [bs x num_patch], representing flow_type, station, weekday, or time_in_day
        """
        z = self.W_P(z)                                                          # z: [bs x num_patch x d_model]
        z = self.dropout(z + self.W_pos)

        # feature encoding
        for i, feature_eb in enumerate(self.feature_eb):
            z += feature_eb(features[i])                            # z: [bs x num_patch x d_model]

        # Encoder
        z = self.encoder(z)                                                      # z: [bs x num_patch x d_model]
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
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        for mod in self.layers: output = mod(output)
        return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0., dropout=0., bias=True,
                activation="gelu", pre_norm=False):
        """pre_norm: if True, apply normalization before residual and multi-head attention."""
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
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
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        elif "layer" in norm.lower():
            self.norm_ffn = nn.LayerNorm(d_model)
        else:
            raise NotImplementedError(f"Normalization {norm} not implemented")

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return src


class PredictionHead(nn.Module):
    def __init__(self, d_model, num_patch, forecast_len, out_dim=1, head_dropout=0):
        super().__init__()
        head_dim = num_patch*d_model
        self.forecast_len = forecast_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_list = nn.ModuleList([nn.Linear(head_dim, forecast_len) for i in range(out_dim)])
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x num_patch x d_model]
        output: [bs x forecast_len x out_dim]
        """
        y = torch.empty(x.shape[0], self.forecast_len, len(self.linear_list), device=x.device)  # [bs x forecast_len x out_dim]
        x = self.flatten(x)     # x: [bs x (num_patch * d_model)]
        x = self.dropout(x)
        for i, linear in enumerate(self.linear_list):
            y[:, :, i] = linear(x)
        return y


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x num_patch x d_model]
        output: tensor [bs x num_patch x patch_len]
        """
        x = self.linear( self.dropout(x) )      # [bs x num_patch x patch_len]
        return x


class MQRnnHead(nn.Module):
    pass
