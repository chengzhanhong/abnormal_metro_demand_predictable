# The transformer models with cache design
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np
from .basics import get_activation_fn, Transpose

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0., proj_dropout=0., qkv_bias=True):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)

        # Project output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output, attn_weights

   
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017)"""

    def __init__(self, d_model, n_heads, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len], indicates which elements of the key sequence should be ignored to deal
                              with when the key sequence has variable length.
            attn_mask       : [1 x seq_len x seq_len] mask out certain elements of the sequence
                            (e.g., to prevent the model from attending to future information when predicting the next time step)
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        return output, attn_weights



# Cell
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                             attn_dropout=attn_dropout, dropout=dropout,
                                                             activation=activation,
                                                             pre_norm=pre_norm, store_attn=store_attn) for i in
                                     range(n_layers)])
        self.previous_cache = None
    def forward(self, src: Tensor, attn_mask=None, cache=False):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Retrieve the cache if available
        if cache and (self.previous_cache is not None):
            src_kv = torch.cat([self.previous_cache, src], dim=1)
            self.previous_cache = src_kv
        else:
            src_kv = src
            if cache:
                self.previous_cache = src

        # Forward pass through the layers using cache if available
        for mod in self.layers:
            src = mod(src, src_kv, attn_mask, cache=cache)
            if cache and (self.previous_cache is not None):
                src_kv = mod.src_kv
            else:  # if cache is False or the first time to cache
                src_kv = src

        return src


class TransformerEncoderLayer(nn.Module):
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
        self.src_kv = None
        self.attn = None

    def forward(self, src: Tensor, src_kv:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None, cache=False):
        """
        src: tensor [bs x q_len_current x d_model] if src_kv is None else [bs x q_len_full x d_model]
        src_kv: tensor [bs x q_len_full x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
            if src_kv is not None:
                src_kv = self.norm_attn(src_kv)
        if src_kv is None:
            src_kv = src

        ## Multi-Head attention
        src2, attn = self.self_attn(src, src_kv, src_kv, attn_mask=attn_mask)
        # attn: [bs x n_heads x q_len x seq_len]

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

        # Store attention weights
        if self.store_attn:
            if (self.attn is None) or (self.src_kv is None):
                self.attn = attn
            else:
                self.attn = torch.cat([self.attn, attn], dim=-1)

        # cache if requested
        if cache:
            if self.src_kv is None:
                self.src_kv = src
            else:
                self.src_kv = torch.cat([self.src_kv, src], dim=1)

        return src
