import os
import numpy as np
import torch
import argparse
import pandas as pd
from matplotlib import pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)
else:
    print('No GPU available, using the CPU instead.')

parser = argparse.ArgumentParser()
args = parser.parse_args([])
args.torch_int = torch.int32
args.default_int = 'int32'
torch.set_default_dtype(torch.float32)
args.default_float = 'float32'

def get_wandb_args(path):
    import wandb
    api = wandb.Api()
    run = api.run(path)
    args = argparse.Namespace(**run.config)
    exec('args.torch_int = ' + args.torch_int)

    if args.default_float == 'float32':
        torch.set_default_dtype(torch.float32)
    elif args.default_float == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError('default float type not supported')

    args.wandb_id = run.id
    args.name = run.name
    return args

def get_num_embedding(args, train_loader):
    num_embeds = []
    if args.station_eb:
        num_embeds.append(train_loader.dataset.num_station)
    if args.weekday_eb:
        num_embeds.append(train_loader.dataset.num_weekday)
    if args.time_eb:
        num_embeds.append(train_loader.dataset.num_time_in_day)
    num_embeds = tuple(num_embeds)
    return num_embeds


def get_attn_mask(num_patch, num_target_patch, type='time', device='cpu'):
    assert type in ['strict', 'boarding', 'cross', 'none', None, 'no_alighting', 'time'], 'type should be one of strict, boarding, cross.'
    if type == 'none' or type is None:
        return None
    num_boarding = num_patch + num_target_patch
    aa = torch.ones(num_patch, num_patch, dtype=torch.bool)  # alighting to alighting
    bb = torch.ones(num_boarding, num_boarding, dtype=torch.bool)  # boarding to boarding
    ab = torch.ones(num_patch, num_boarding, dtype=torch.bool)  # alighting to boarding
    ba = torch.ones(num_boarding, num_patch, dtype=torch.bool)  # boarding to alighting
    ab = ab.triu(1)
    ba = ba.triu(1)
    if type == 'strict':
        aa = aa.triu(1)
    elif type == 'boarding' or type == 'cross':
        aa.zero_()

    if type == 'strict' or type=='boarding':
        bb = bb.triu(1)
    elif type == 'cross':
        bb.zero_()

    if type == 'no_alighting':
        ab = torch.ones(num_patch, num_boarding, dtype=torch.bool)  # alighting to boarding
        ba = torch.ones(num_boarding, num_patch, dtype=torch.bool)  # boarding to alighting
        bb = bb.triu(1)

    if type == 'time':
        bb = bb.triu(1)
        return bb.to(device)

    attn_mask = torch.cat([torch.cat([aa, ab], dim=1), torch.cat([ba, bb], dim=1)], dim=0)
    return attn_mask.to(device)


def get_time_x(start, end, patch_len, num_points):
    eps = 1/patch_len/2
    start = start - 0.5 + eps
    end = end + 0.5 - eps
    x = np.linspace(start, end, num_points)
    return x


def exam_attention(t, s, model, dataset, args, loc=75, layer=2):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t1 = pd.Timestamp(t)
    (x, y), data_piece = dataset.get_data_from_ts(t1, s)

    x = [xx.unsqueeze(0) for xx in x]
    x = [xx.to(device) for xx in x]
    model.eval()
    y_hat = model(x)   # (1, num_fcst, patch_len, output_dim)
    y_hat = y_hat.detach().cpu().numpy().reshape(-1, model.output_dim)  # (num_fcst*patch_len, output_dim)

    if args.loss in {'quantile3', 'quantile5'}:
        y_mean = y_hat[:, 0]
        y_lb = y_hat[:, 1]
        y_ub = y_hat[:, 2]
    elif args.loss == 'gaussian_nll':
        y_mean = y_hat[:, 0]
        y_lb = y_hat[:, 0] - y_hat[:, 1]
        y_ub = y_hat[:, 0] + y_hat[:, 1]
    else:
        y_mean = y_hat
        y_lb = None
        y_ub = None

    attns = model.backbone.encoder.layers[layer].attn.squeeze(0).detach().cpu().numpy()

    # loc = 75  # The location of the patch
    attn_mats = [attns[head][[loc], :] for head in range(args.n_heads)]
    attn_mats = np.concatenate(attn_mats, axis=0)

    # A figure with two subplots, gap between rows is 0
    fig, axes = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0})
    num_patch = args.num_patch
    patch_len = args.patch_len
    context_len = args.context_len
    num_target_patch = args.num_target_patch
    axes[1].imshow(attn_mats, cmap='gray', aspect='auto')
    axes[0].plot(get_time_x(0, num_patch-1, patch_len, context_len),
                 data_piece['outflow'].values[:args.context_len], label='Real alighting', color='C1')
    axes[0].plot(get_time_x(num_patch, attn_mats.shape[1]-1, patch_len, dataset.sample_len),
                 data_piece['inflow'].values, label='Real boarding', color='C0')
    # a vertical line at the location of the target patch
    axes[0].axvline(x=loc, color='red', linestyle='--')
    # set title
    axes[0].set_title(f'{t} at station {s}')

    # plot the forecasted values
    x_time = get_time_x(2*num_patch, attn_mats.shape[1]-1, patch_len, num_target_patch*patch_len)
    axes[0].plot(x_time, y_mean, label='Prediction', color='C2')
    # fill the confidence interval
    if y_lb is not None:
        axes[0].fill_between(x_time, y_lb, y_ub, color='C2', alpha=0.2, interpolate=True)
    axes[0].legend()

    return fig


def get_loc_scale(train_data, standardization):
    if standardization == 'zscore':
        x_loc = train_data.groupby('station')['inflow'].mean().values
        x_scale = train_data.groupby('station')['inflow'].std().values
    elif standardization == 'meanscale':
        x_loc = torch.zeros(train_data['station'].nunique(), dtype=torch.float32)
        x_scale = train_data.groupby('station')['inflow'].mean().values
    elif standardization == 'minmax':
        x_loc = torch.zeros(train_data['station'].nunique(), dtype=torch.float32)
        x_scale = (train_data.groupby('station')['inflow'].max() - train_data.groupby('station')['inflow'].min()).values
    elif standardization == 'none':
        x_loc = torch.zeros(train_data['station'].nunique(), dtype=torch.float32)
        x_scale = torch.ones(train_data['station'].nunique(), dtype=torch.float32)
    else:
        raise ValueError('standardization not supported')
    return x_loc, x_scale

