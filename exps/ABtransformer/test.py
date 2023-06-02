#%%
import time
script_start_time = time.time()

import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from datasets.datautils import *
from models.MetroTransformer import MetroTransformer
from utilities.lr_finder import LRFinder
from utilities.basics import *
from utilities.loss import *
import argparse

#%% Define the default arguments
parser = argparse.ArgumentParser()
# General
parser.add_argument('--default_float', type=str, default='float32', help='default float type')
parser.add_argument('--default_int', type=str, default='int32', help='default int type')
# parser.add_argument('--task', type=str, default='supervised', help='one of supervised, unsupervised, finetune')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--max_run_time', type=int, default=10, help='maximum running time in hours')

# Dataset and dataloader
parser.add_argument('--dset', type=str, default='guangzhou', help='dataset name, guangzhou or nyc or seoul')
parser.add_argument('--subsample', type=bool, default=False, help='Whether to subsample the dataset for quick test')
parser.add_argument('--datatype', type=str, default='seq', help='one of seq, concat')
parser.add_argument('--data_mask_method', type=str, default='target', help='one of (target, both)')
parser.add_argument('--data_mask_ratio', type=float, default=0.2, help='the ratio of masked data in context')
parser.add_argument('--train_r', type=float, default=0.8, help='the ratio of training data')
parser.add_argument('--val_r', type=float, default=0.1, help='the ratio of validation data')
parser.add_argument('--test_r', type=float, default=0.1, help='the ratio of test data')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--flow_diff_r', type=float, default=0.4, help='the maximum possible ratio of in-out flow '
                                                                   'difference of a day, used to remove outliers')
parser.add_argument('--standardization', type=str, default='iqr', help='one of (iqr, zscore, minmax)')


# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=8, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Transformer MLP dimension')
parser.add_argument('--attn_dropout', type=float, default=0.2, help='Transformer attention dropout')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
parser.add_argument('--pre_norm', type=bool, default=False, help='whether to apply normalization before attention')
parser.add_argument('--activation', type=str, default='gelu', help='activation function in Transformer')
parser.add_argument('--store_attn', type=bool, default=True, help='whether to store attention weights')
parser.add_argument('--norm', type=str, default='LayerNorm', help='normalization layer, one of (BatchNorm, LayerNorm)')
parser.add_argument('--max_lr', type=float, default=1e-3, help='maximum learning rate for one cycle policy')

# positional encoding and feature embedding
parser.add_argument('--pe', type=str, default='zeros', help='type of position encoding (zeros, sincos, or none)')
parser.add_argument('--learn_pe', type=bool, default=True, help='learn position encoding')
parser.add_argument('--flow_eb', type=bool, default=True, help='whether to use flow embedding (inflow and outflow) or not.')
parser.add_argument('--station_eb', type=bool, default=True, help='whether to use station embedding or not.')
parser.add_argument('--weekday_eb', type=bool, default=True, help='whether to use weekday embedding or not.')
parser.add_argument('--time_eb', type=bool, default=True, help='whether to use time embedding or not.')
parser.add_argument('--attn_mask_type', type=str, default='none', help='the type of attention mask, one of none, '
                                                                       'strict, boarding, cross')

# Optimization args
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--loss', type=str, default='quantile1', help='loss function, one of '
                                                                  '(quantile1, quantile3, quantile5, rmse)')

args = parser.parse_args([])

#%% Setups
# Set default float and int types
if args.default_float == 'float32':
    torch.set_default_dtype(torch.float32)
elif args.default_float == 'float64':
    torch.set_default_dtype(torch.float64)
else:
    raise ValueError('default float type not supported')

if args.default_int == 'int32':
    args.torch_int = torch.int32
elif args.default_int == 'int64':
    args.torch_int = torch.int64
else:
    raise ValueError('default int type not supported')

# Set device to GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)
else:
    print('No GPU available, using the CPU instead.')

#%% Prepare data
args.dset = 'guangzhou'
args.subsample = False
args.n_epochs = 2
args.d_model = 128
args.n_heads = 8
args.n_layers = 3
args.loss = 'gaussian_nll'  # one of (quantile1, quantile3, quantile5, rmse, mae, gaussian_nll)
args.max_lr = 0.001
args.standardization = 'zscore'
args.attn_mask_type = 'none'
args.pre_norm = True
args.pe = 'zeros'
args.learn_pe = True
args.anneal_strategy = 'linear'
args.data_mask_method = 'both'
data_mask_method = args.data_mask_method

data_info = data_infos[args.dset]
vars(args).update(data_info)
args.context_len = get_context_len(args.patch_len, args.num_patch, args.stride)
args.num_target_patch = args.target_len // args.patch_len
print('args:', args)
# Set random seed
reset_random_seeds(args.seed)

data = read_data(args)
data = detect_anomaly(data, args)
dataset = MetroDataset_total(data, args, datatype=args.datatype)
train_dataset = dataset.TrainDataset
val_dataset = dataset.ValDataset
test_dataset = dataset.TestDataset
train_dataset.mask_method = args.data_mask_method; train_dataset.mask_ratio = args.data_mask_ratio
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_data = train_dataset.data.loc[train_dataset.f_index, :]
if args.standardization == 'iqr':
    x_loc = train_data.groupby('station')['inflow'].median().values
    x_scale = (train_data.groupby('station')['inflow'].quantile(0.75) - train_data.groupby('station')['inflow'].quantile(0.25)).values
elif args.standardization == 'zscore':
    x_loc = train_data.groupby('station')['inflow'].mean().values
    x_scale = train_data.groupby('station')['inflow'].std().values
elif args.standardization == 'minmax':
    x_loc = train_data.groupby('station')['inflow'].min().values
    x_scale = (train_data.groupby('station')['inflow'].max() - train_data.groupby('station')['inflow'].min()).values
else:
    raise ValueError('standardization not supported')
x_loc = torch.asarray(x_loc, dtype=torch.float32).to(device)
x_scale = torch.asarray(x_scale, dtype=torch.float32).to(device)


# %% Examine models

# %% prepare data
# args = get_wandb_args('deepzhanhong/MetroTransformer/fj50u75p')
# args = get_wandb_args('deepzhanhong/MetroTransformer/utadtdug')
args = get_wandb_args('deepzhanhong/MetroTransformer/cx0kvkrx')
args.data_path = '..//..//..//data/GuangzhouMetro//'
args.datatype = 'seq'
# Set random seed
reset_random_seeds(args.seed)

data = read_data(args)
data = detect_anomaly(data, args)
dataset = MetroDataset_total(data, args, datatype=args.datatype)
train_dataset = dataset.TrainDataset
val_dataset = dataset.ValDataset
test_dataset = dataset.TestDataset
train_dataset.mask_method = args.data_mask_method; train_dataset.mask_ratio = args.data_mask_ratio
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_data = train_dataset.data.loc[train_dataset.f_index, :]
if args.standardization == 'iqr':
    x_loc = train_data.groupby('station')['inflow'].median().values
    x_scale = (train_data.groupby('station')['inflow'].quantile(0.75) - train_data.groupby('station')['inflow'].quantile(0.25)).values
elif args.standardization == 'zscore':
    x_loc = train_data.groupby('station')['inflow'].mean().values
    x_scale = train_data.groupby('station')['inflow'].std().values
elif args.standardization == 'minmax':
    x_loc = train_data.groupby('station')['inflow'].min().values
    x_scale = (train_data.groupby('station')['inflow'].max() - train_data.groupby('station')['inflow'].min()).values
else:
    raise ValueError('standardization not supported')
x_loc = torch.asarray(x_loc, dtype=torch.float32).to(device)
x_scale = torch.asarray(x_scale, dtype=torch.float32).to(device)

num_embeds = []
if args.flow_eb:
    num_embeds.append(train_loader.dataset.num_flow_type)
if args.station_eb:
    num_embeds.append(train_loader.dataset.num_station)
if args.weekday_eb:
    num_embeds.append(train_loader.dataset.num_weekday)
if args.time_eb:
    num_embeds.append(train_loader.dataset.num_time_in_day)
num_embeds = tuple(num_embeds)
args.num_embeds = num_embeds
attn_mask = get_attn_mask(args.num_patch, args.num_target_patch, args.attn_mask_type, device=device)
# Load the model
model = MetroTransformer(x_loc=x_loc, x_scale=x_scale, attn_mask=attn_mask, **args.__dict__)
model.load_state_dict(torch.load(f'log//{args.name}.pth', map_location=torch.device('cpu')))

# %% Examine the attention
exam_attention(t='2017-07-22 20:00:00', s=33, model=model, dataset=dataset, args=args, layer=2, loc=76)
exam_attention(t='2017-09-23 20:00:00', s=33, model=model, dataset=test_dataset, args=args, layer=2, loc=77)
exam_attention(t='2017-07-15 15:30:00', s=110, model=model, dataset=train_dataset, args=args, layer=2, loc=77)
exam_attention(t='2017-07-29 20:00:00', s=110, model=model, dataset=train_dataset, args=args, layer=2, loc=76)
exam_attention(t='2017-08-05 20:00:00', s=110, model=model, dataset=train_dataset, args=args, layer=2, loc=76)
exam_attention(t='2017-09-23 20:00:00', s=110, model=model, dataset=test_dataset, args=args, layer=2, loc=76)

exam_attention(t='2017-08-01 20:00:00', s=13, model=model, dataset=train_dataset, args=args, loc=76)
exam_attention(t='2017-09-08 20:00:00', s=13, model=model, dataset=train_dataset, args=args, loc=75)
exam_attention(t='2017-09-16 20:00:00', s=13, model=model, dataset=val_dataset, args=args, loc=75)

# exam_attention(t='2017-09-01 16:00:00', s = 18, model=model, dataset=train_dataset, args=args, loc=75)

# exam_attention(t='2017-07-15 20:00:00', s = 33, model=model, dataset=train_dataset, args=args, layer=0)
exam_attention(t='2017-07-29 20:00:00', s=33, model=model, dataset=train_dataset, args=args, layer=2, loc=77)
exam_attention(t='2017-07-26 18:00:00', s=33, model=model, dataset=train_dataset, args=args)
exam_attention(t='2017-09-16 20:00:00', s=33, model=model, dataset=val_dataset, args=args)
exam_attention(t='2017-09-23 20:00:00', s=33, model=model, dataset=test_dataset, args=args, layer=2, loc=77)
exam_attention(t='2017-09-12 18:00:00', s=51, model=model, dataset=val_dataset, args=args)

#%% Examine the performance in the test set
test_mae, test_rmse = evaluate(model, test_loader, device='cpu')

# %% check the low-rankness of the station embedding
weight = model.backbone.feature_eb[-2].weight.detach().numpy()
[u, s, v] = np.linalg.svd(weight)
plt.plot(s)

a = np.random.randn(100, 10)
b = np.random.randn(10, 100)
c = np.matmul(a, b)
plt.hist(a.flatten(), bins=100)
plt.hist(c.flatten(), bins=100)
