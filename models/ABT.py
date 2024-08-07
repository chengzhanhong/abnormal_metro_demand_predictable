from .pos_encoding import *
from .basics import *
from .transformer import *


class ABTransformer(nn.Module):
    def __init__(self, patch_len: int, num_patch: int, num_target_patch, num_embeds: tuple = (2, 159, 7, 217),
                 n_layers: int = 3, d_model=128, n_heads=16, d_ff: int = 256,
                 norm: str = 'LayerNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu",
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, head_dropout=0,
                 attn_mask=None, x_loc=None, x_scale=None, input_type='number', head_type='RMSE',
                 input_emb_size=8, num_bins=None, bin_edges=None, top_p=0.9, b=20000, **kwargs):
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
        self.flow_type_eb = nn.Parameter(torch.tensor([[0, 1]], dtype=torch.long), requires_grad=False)

        # Standardization
        self.standardization = Standardization(self.x_loc, self.x_scale, input_type, head_type)

        # Backbone
        self.backbone = MetroEncoder(num_patch=num_patch, patch_len=patch_len, num_target_patch=num_target_patch,
                                     num_embeds=(2,)+tuple(num_embeds), n_layers=n_layers, d_model=d_model,
                                     n_heads=n_heads, d_ff=d_ff, attn_dropout=attn_dropout,
                                     dropout=dropout, act=act, pre_norm=pre_norm, store_attn=store_attn,
                                     norm=norm, pe=pe, learn_pe=learn_pe,
                                     num_bins=num_bins, input_emb_size=input_emb_size, input_type=input_type)

        # The output head part
        if self.head_type == 'CrossEntropy':
            self.bin2num_map = torch.nn.Parameter(torch.tensor([bin2num(x, bin_edges) for x in range(num_bins)], dtype=torch.float32).squeeze())
            self.bin_edges = torch.nn.Parameter(torch.from_numpy(bin_edges), requires_grad=False)
        else:
            self.bin2num_map = None
            self.bin_edges = None

        self.head = head_dic[head_type](d_model, patch_len, head_dropout, num_bins=num_bins)
        self.mean = mean_dic[head_type](bin2num_map=self.bin2num_map, bin_edges=self.bin_edges, return_type=self.input_type, b=b)
        self.sample = sample_dic[head_type](bin_edges=self.bin_edges, top_p=top_p, return_type=self.input_type, b=b)

    def forward(self, x, method='param'):
        """
        x: tuple of flow tensor [bs x num_patch x 2*patch_len] and feature tensors of [bs x num_patch].
        """
        # Prepare input
        n_patch = x[0].size(1)
        z = torch.cat((x[0][:, :, :self.patch_len], x[0][:, :, -self.patch_len:]), dim=1)  # [bs x 2*num_patch x patch_len]
        attn_mask = self.attn_mask[:n_patch, :n_patch].repeat(2,2)  # [2*num_patch x 2*num_patch]
        flow_type_eb = self.flow_type_eb.repeat_interleave(n_patch, dim=1)  # [1 x 2*num_patch]
        features = [flow_type_eb] + [f.repeat(1, 2) for f in x[1:]]
        stations = x[1][:, 0].long()

        # Model
        z = self.standardization(z, stations, 'norm')
        z = self.backbone(z, features, attn_mask)  # z: [bs x 2*num_patch x d_model]
        z = self.head(z)
        z = self.standardization(z, stations, 'denorm')  # z: [bs x 2*num_patch x patch_len x if any]
        z =  tuple([torch.cat((z[i][:,:n_patch], z[i][:,-n_patch:]), dim=2) for i in range(len(z))]) # [bs x num_patch x patch_len*2 x if any]

        if method=='param':
            return z
        elif method=='mean':
            return self.mean(*z)
        elif method=='sample':
            return self.sample(*z)
        else:
            raise ValueError("method should be 'param', 'mean', or 'sample'.")


    def forecast(self, x, method='mean'):
        """
        Auto-regressive forecasting.
        x: tuple of flow tensor [bs x num_patch x patch_len] and feature tensors of [bs x num_patch x 1].
        """
        n_target = self.num_target_patch  # number of patches to predict
        n_input = x[0].shape[1] - n_target + 1
        y = x[0][:, :n_input, :].clone()  # The input and also the prediction
        features = x[1:]

        self.eval()
        with torch.no_grad():
            for i in range(n_target):
                xx = [y] + [feature[:, :n_input+i] for feature in features]
                y_new = self(xx, method=method)
                y = torch.cat([y, y_new[:, -1:, :]], dim=1)

        y = y[:, -n_target:, :]
        if self.head_type == 'CrossEntropy' and self.input_type == 'bins':
            y = self.bin2num_map[y]

        return y

    def forecast_samples(self, x, n=100):
        """Autoregressive forecasting in the test phase, draw n samples
        """
        result = []
        with torch.no_grad():
            for i in range(n):
                result.append(self.forecast(x=x, method='sample'))
        return torch.stack(result, dim=0)


class MetroEncoder(nn.Module):
    def __init__(self, num_patch, num_target_patch, patch_len, num_embeds, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 pre_norm=False, pe='zeros', learn_pe=True, num_bins=None, input_emb_size=None, input_type='number'):
        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if input_type == 'number':
            self.W_P = nn.Linear(patch_len, d_model)
        elif input_type == 'bins':
            self.W_P = nn.Sequential(nn.Embedding(num_bins, input_emb_size), FlattenLastTwoDims(), nn.Dropout(dropout),
                                     nn.Linear(input_emb_size, d_model))

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_target_patch+num_patch-1, d_model)

        # Flow_type, station, weekday, or time_in_day encoding
        self.feature_eb = nn.ModuleList([nn.Embedding(num_embeds[i], d_model) for i in range(len(num_embeds))])

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                          pre_norm=pre_norm, activation=act, n_layers=n_layers, store_attn=store_attn)

    def forward(self, z, features, attn_mask=None) -> Tensor:
        """
        z: tensor [bs x 2*num_patch x patch_len]
        features: tuple of tensor [bs x 2*num_patch], representing flow_type, station, weekday, time_in_day
        """
        z_len = int(z.shape[1]/2)
        z = self.W_P(z)
        z = z + self.W_pos[:z_len, :].repeat(2, 1)

        # feature encoding
        for i, feature_eb in enumerate(self.feature_eb):
            z += feature_eb(features[i])  # z: [bs x num_patch x d_model]

        z = self.dropout(z)
        # Encoder
        z = self.encoder(z, attn_mask)  # z: [bs x num_patch x d_model]
        return z

