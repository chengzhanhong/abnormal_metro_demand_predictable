# Notice I used the same affine transform for both the inflow and outflow in ABT, which restricts the output space
# Here in ABT2, I use different affine transforms for inflow and outflow, and remove the flow type embedding
from .pos_encoding import *
from .basics import *
from .transformer import *


class ABTransformer(nn.Module):
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
                 attn_mask=None, x_loc=None, x_scale=None, head='RMSE',**kwargs):
        """
        Parameters:
            num_embeds: tuple of number of embeddings for station, weekday, or time_in_day
            d_ff: dimension of the inner feed-forward layer
            pe: type of positional encoding initialization, "zeros", "sincos", or None
            learn_pe: whether to learn positional encoding
            revin: whether to use reversible instance normalization
            ABflow: whether to use AB flow model
        """
        super().__init__()
        self.patch_len = patch_len
        self.num_patch = num_patch  # number of patches of input outflow
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

        self.head = head_dic[head](d_model, patch_len, head_dropout)
        self.mean = mean_dic[head]()
        self.sample = sample_dic[head]()
        # Standardization
        self.standardization = Standardization(self.x_loc, self.x_scale, head)
        self.softplus = nn.Softplus()

    def forward(self, x, method='param'):
        """
        x: tuple of flow tensor [bs x num_patch x 2*patch_len] and feature tensors of [bs x num_patch].
        """
        # Prepare input
        n_patch = x[0].size(1)
        z = x[0]  # [bs x num_patch x 2*patch_len]
        attn_mask = self.attn_mask[:n_patch, :n_patch].repeat(2,2)  # [2*num_patch x 2*num_patch]
        features = x[1:]
        stations = x[1][:, 0].long()

        # Model
        z = self.standardization(z, stations, 'norm')
        z = self.backbone(z, features, attn_mask)  # z: [bs x 2*num_patch x d_model]
        z = self.head(z)
        z = self.standardization(z, stations, 'denorm')  # z: [bs x 2*num_patch x patch_len]
        z =  tuple([torch.cat((z[i][:,:n_patch,:], z[i][:,-n_patch:,:]), dim=-1) for i in range(len(z))]) # [bs x num_patch x patch_len*2]

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

        return y[:, -n_target:, :]

class MetroEncoder(nn.Module):
    def __init__(self, num_patch, num_target_patch, patch_len, num_embeds, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 pre_norm=False, pe='zeros', learn_pe=True):
        super().__init__()
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model

        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.W_P_in = nn.Linear(patch_len, d_model)
        self.W_P_out = nn.Linear(patch_len, d_model)

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
        z: tensor [bs x num_patch x 2*patch_len]
        features: tuple of tensor [bs x 2*num_patch], representing flow_type, station, weekday, time_in_day
        """
        z_len = int(z.shape[1])
        z_in = self.W_P_in(z[:, :, :self.patch_len])
        z_out = self.W_P_out(z[:, :, -self.patch_len:])
        z = torch.cat((z_in, z_out), dim=1)

        # Positional encoding
        z = z + self.W_pos[:z_len, :].repeat(2, 1)

        # feature encoding
        features = [f.repeat(1, 2) for f in features]
        for i, feature_eb in enumerate(self.feature_eb):
            z += feature_eb(features[i])  # z: [bs x num_patch x d_model]

        z = self.dropout(z)
        # Encoder
        z = self.encoder(z, attn_mask)  # z: [bs x num_patch x d_model]
        return z
