import os
import random
import numpy as np
import torch
import argparse
import pandas as pd
from matplotlib import pyplot as plt

def reset_random_seeds(n=1):
    os.environ['PYTHONHASHSEED'] = str(n)
    # tf.random.set_seed(n)
    torch.random.manual_seed(n)
    np.random.seed(n)
    random.seed(n)

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


def get_attn_mask(num_patch, num_target_patch, type='strict', device='cpu'):
    assert type in ['strict', 'boarding', 'cross', 'none', None], 'type should be one of strict, boarding, cross.'
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

    attn_mask = torch.cat([torch.cat([aa, ab], dim=1), torch.cat([ba, bb], dim=1)], dim=0)
    return attn_mask.to(device)


def exam_result(t, s, model, dataset, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t = pd.Timestamp(t)
    (x, y), data_piece = dataset.get_data_from_ts(t, s)
    fcst_loc = x[-1].detach().numpy().reshape(-1,1) - args.num_patch
    fcst_loc = fcst_loc*args.patch_len + np.arange(args.patch_len).reshape(1,-1)
    fcst_loc = fcst_loc.reshape(-1)

    x = [xx.unsqueeze(0) for xx in x]
    x = [xx.to(device) for xx in x]
    model.eval()
    y_hat = model(x)   # (1, num_fcst, patch_len)
    y_hat = y_hat.detach().cpu().numpy().reshape(-1)

    fig, ax = plt.subplots()
    ax.plot(data_piece['time'].values, data_piece['inflow'].values, label='Real boarding', color='C0')
    ax.plot(data_piece['time'].values[:args.context_len],
            data_piece['outflow'].values[:args.context_len],
            label='Real alighting', color='C1')
    ax.plot(data_piece['time'].values[fcst_loc], y_hat, '-o', label='Prediction', color='C2')
    ax.set_xlabel('Time')
    # rotate xticks labels
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.legend()
    plt.show()
    return fig

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
    y_hat = model(x)   # (1, num_fcst, patch_len)
    y_hat = y_hat.detach().cpu().numpy().reshape(-1)

    attns = model.backbone.encoder.layers[layer].attn.squeeze(0).detach().cpu().numpy()
    #     for head in range(args.n_heads):
    #         attn = attns[head]
    # fig, ax = plt.subplots()
    # ax.imshow(attn, cmap='gray', vmax=1, vmin=0)

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
    axes[0].plot(get_time_x(2*num_patch, attn_mats.shape[1]-1, patch_len, num_target_patch*patch_len),
                 y_hat, label='Prediction', color='C2')
    # a vertical line at the location of the target patch
    axes[0].axvline(x=loc, color='red', linestyle='--')
    # set title
    axes[0].set_title(f'{t} at station {s}')
    return fig

