#%%
# In this experment, the inputs are concat of inflow and outflow, each inflow element consists of three 10min flow data.
# Prior att Mask is set to None
# Three flow types setting for patches: complete in-out flow batch, inflow masked batch, inflow to predict batch
import time
script_start_time = time.time()

import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from datasets.datautils import *
from NBEATS import NeuralBeats
from utilities.basics import *
from utilities.loss import *
import argparse

#%% Define the default arguments
parser = argparse.ArgumentParser()
# General
parser.add_argument('--default_float', type=str, default='float32', help='default float type')
parser.add_argument('--default_int', type=str, default='int32', help='default int type')
parser.add_argument('--task', type=str, default='supervised', help='one of supervised, unsupervised, finetune')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--max_run_time', type=int, default=10, help='maximum running time in hours')

# Dataset and dataloader
parser.add_argument('--dset', type=str, default='seoul', help='dataset name seoul')
parser.add_argument('--subsample', type=bool, default=False, help='Whether to subsample the dataset for quick test')
parser.add_argument('--t_resolution', type=str, default='10T', help='the time resolution for resampling')
parser.add_argument('--patch_len', type=int, default=6, help='patch length')
parser.add_argument('--stride', type=int, default=6, help='stride between patch')
parser.add_argument('--num_patch', type=int, default=18, help='number of patches for boarding/alighting flow')
parser.add_argument('--target_len', type=int, default=36, help='forecast horizon for supervised learning, '
                                                               'should be a multiple of stride/patch_len')
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
args.num_target_patch = args.target_len // args.patch_len

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
if args.dset == "guangzhou":
    args.data_path = '../../../data/GuangzhouMetro//'
if args.dset == "seoul":
    args.data_path = '../../../data/SeoulMetro//'
args.subsample = False
args.n_epochs = 40
args.d_model = 128
args.n_heads = 8
args.n_layers = 3
args.patch_len = 3
args.stride = 3
args.num_patch = 36
args.loss = 'rmse'
args.max_lr = 0.001
args.standardization = 'zscore'
args.attn_mask_type = 'strict'
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
train_data, val_data, test_data = split_train_val_test(data, args)


#%% Main experiments
stations = data['station'].unique()
train_data = train_data[train_data['station'] == 1]
test_data = test_data[test_data['station'] == 1]
train_inflow = train_data['inflow'].to_numpy().reshape(-1, 1)
test_inflow = train_data['inflow'].to_numpy().reshape(-1, 1)
model = NeuralBeats(data=train_inflow, forecast_length=36)
model.fit(epoch=25,optimiser=torch.optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999),
                                               eps=1e-07, weight_decay=0.01, amsgrad=False),plot=False, verbose=True)
forecast = model.predict(predict_data=test_inflow)