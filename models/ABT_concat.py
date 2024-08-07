from .pos_encoding import *
from .basics import *
from .transformer import *


class ABTransformer_concat(nn.Module):
    def __init__(self, patch_len: int, num_patch: int, num_target_patch, num_embeds: tuple = (2, 159, 7, 217),
                 n_layers: int = 3, d_model=128, n_heads=16, d_ff: int = 256,
                 norm: str = 'LayerNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0,
                 attn_mask=None, x_loc=None, x_scale=None, input_type='number', head_type='RMSE',
                 input_emb_size=8, num_bins=None, bin_edges=None, top_p=0.9, n_mixture=2,
                 forecast_target='both', **kwargs):
        super().__init__()
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.num_target_patch = num_target_patch
        self.attn_mask = attn_mask
        self.input_type = input_type
        self.head_type = head_type
        self.num_bins = num_bins
        self.num_embeds = num_embeds
        self.d_model = d_model
        self.n_layers = n_layers
        self.model_name = 'ABTransformer_concat'
        self.forecast_target = forecast_target
        if forecast_target == 'both':
            factor = 2
        elif forecast_target in {'inflow', 'outflow'}:
            factor = 1
        else:
            raise ValueError("forecast_target should be 'both', 'inflow', or 'outflow'.")

        # Standardization
        self.standardization = Standardization(self.x_loc, self.x_scale, input_type, head_type)

        # Backbone
        self.backbone = MetroEncoder(num_patch=num_patch, patch_len=patch_len, num_target_patch=num_target_patch,
                                     num_embeds=num_embeds, n_layers=n_layers, d_model=d_model,
                                     n_heads=n_heads, d_ff=d_ff, attn_dropout=attn_dropout,
                                     dropout=dropout, act=act, pre_norm=pre_norm, store_attn=store_attn,
                                     norm=norm, pe=pe, learn_pe=learn_pe,
                                     num_bins=num_bins, input_emb_size=input_emb_size, input_type=input_type,
                                     forecast_target=forecast_target)

        # The output head part
        if self.head_type == 'CrossEntropy':
            self.bin2num_map = torch.nn.Parameter(torch.tensor([bin2num(x, bin_edges) for x in range(num_bins)], dtype=torch.float32).squeeze())
            self.bin_edges = torch.nn.Parameter(torch.from_numpy(bin_edges), requires_grad=False)
        else:
            self.bin2num_map = None
            self.bin_edges = None

        self.head = head_dic[head_type](d_model, patch_len * factor, head_dropout, num_bins=num_bins, n_mixture=n_mixture)
        self.mean = mean_dic[head_type](bin2num_map=self.bin2num_map, bin_edges=self.bin_edges, return_type=self.input_type)
        self.sample = sample_dic[head_type](bin_edges=self.bin_edges, top_p=top_p, return_type=self.input_type)

    def forward(self, x, method='param', cache=False):
        """
        x: tuple of flow tensor [bs x num_patch x patch_len] and feature tensors of [bs x num_patch x 1].
        """
        n_patch = x[0].size(1)
        if (self.attn_mask is not None) and (n_patch>1):
            attn_mask = self.attn_mask[:n_patch, :n_patch]
        else:
            attn_mask = None # attn_mask is None when n_patch=1 (using cache)

        z = x[0]  # [bs x num_patch x 2*patch_len]
        features = x[1:]
        stations = x[1][:, 0].long()

        z = self.standardization(z, stations, 'norm')
        z = self.backbone(z, features, attn_mask, cache=cache)
        z = self.head(z)
        z = self.standardization(z, stations, 'denorm')  # Only the first output (usually the mean) is denormalized

        if method=='param':
            return z
        elif method=='mean':
            return self.mean(*z)
        elif method=='sample':
            return self.sample(*z)
        else:
            raise ValueError("method should be 'param', 'mean', or 'sample'.")

    def forecast(self, x, method='mean', cache=True):
        """
        Auto-regressive forecasting.
        x: tuple of flow tensor [bs x num_patch x patch_len] and feature tensors of [bs x num_patch x 1].
        """
        n_target = self.num_target_patch  # number of patches to predict
        n_input = x[0].shape[1] - n_target + 1
        features = x[1:]

        result = []
        self.eval()
        with torch.no_grad():
            xx = [x[0][:, :n_input, :]] + [feature[:, :n_input] for feature in features]
            y_new = self(xx, method=method, cache=cache)
            for i in range(n_target-1):
                y_new = y_new[:, [-1], :]
                result.append(y_new)
                xx = [y_new] + [feature[:, [n_input + i]] for feature in features]
                y_new = self(xx, method=method, cache=cache)
        result.append(y_new)
        result = torch.cat(result, dim=1)

        if (self.head_type == 'CrossEntropy') and (self.input_type == 'bins'):
            result = self.bin2num_map[result]
        self.reset_cache()

        return result

    def forecast_samples(self, x, n=100):
        """Autoregressive forecasting in the test phase, draw n samples
        """
        with torch.no_grad():
            if x[0].shape[0] == 1:
                xx = [x[0].repeat(n, 1, 1)] + [feature.repeat(n, 1) for feature in x[1:]]
                result = self.forecast(x=xx, method='sample')
                return result
            else:
                result = []
                for i in range(n):
                    result.append(self.forecast(x=x, method='sample'))
                return torch.stack(result, dim=0)

    def reset_cache(self):
        self.backbone.current_pos = None
        self.backbone.encoder.previous_cache = None
        for layer in self.backbone.encoder.layers:
            layer.src_kv = None


class MetroEncoder(nn.Module):
    def __init__(self, num_patch, num_target_patch, patch_len, num_embeds, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 pre_norm=False, pe='zeros', learn_pe=True, num_bins=None, input_emb_size=None, input_type='number',
                forecast_target='both'):
        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.forecast_target = forecast_target
        if forecast_target == 'both':
            factor = 2
        elif forecast_target in {'inflow', 'outflow'}:
            factor = 1
        else:
            raise ValueError("forecast_target should be 'both', 'inflow', or 'outflow'.")

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if input_type == 'number':
            self.W_P = nn.Linear(patch_len * factor, d_model)
        elif input_type == 'bins':
            self.W_P = nn.Sequential(nn.Embedding(num_bins, input_emb_size), FlattenLastTwoDims(), nn.Dropout(dropout),
                                     nn.Linear(input_emb_size*2, d_model))

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_target_patch+num_patch-1, d_model)

        # Station, weekday, or time_in_day encoding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                          pre_norm=pre_norm, activation=act, n_layers=n_layers, store_attn=store_attn, pe=pe,
                                          encoder_type='ABT_concat')
        self.current_pos = None
        self.pe = pe

    def forward(self, z, features, attn_mask=None, cache=False) -> Tensor:
        """
        z: tensor [bs x num_patch x patch_len]
        features: tuple of tensor [bs x num_patch], representing station, weekday, or time_in_day
        """
        z_len = z.shape[1]
        z = self.W_P(z)  # z: [bs x num_patch x d_model]

        # Determine the positional encoding
        if cache:
            if self.current_pos is None:
                z_slice = slice(0, z_len)
                self.current_pos = z_len
            else:
                z_slice = slice(self.current_pos, self.current_pos+z_len) if z_len>1 else [self.current_pos]
                self.current_pos += z_len
        else:
            self.current_pos = None
            z_slice = slice(0, z_len)

        if self.pe == 'zeros' or self.pe == 'sincos':
            z = z + self.W_pos[z_slice, :]
            # skip None positional encoding, rotary positional embedding is applied in attention layers

        # feature encoding
        for i, feature_eb in enumerate(self.feature_eb):
            z += feature_eb(features[i])  # z: [bs x num_patch x d_model]

        z = self.dropout(z)
        # Encoder
        z = self.encoder(z, attn_mask, cache=cache)  # z: [bs x num_patch x d_model]
        return z
