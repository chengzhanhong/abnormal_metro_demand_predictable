#%%
import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datasets.datautils import *
import torch
import torch.nn as nn
from models.MetroTransformer import MetroTransformer
from models.revin import RevIN
from utilities.lr_finder import LRFinder
from utilities.basics import *
from utilities.loss import *
import argparse
#%% Define the default arguments
parser = argparse.ArgumentParser()
# General
parser.add_argument('--framework', type=str, default='ABtransformer', help='framework name')
parser.add_argument('--default_float', type=str, default='float32', help='default float type')
parser.add_argument('--default_int', type=str, default='int32', help='default int type')
parser.add_argument('--task', type=str, default='supervised', help='one of supervised, unsupervised, finetune')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Dataset and dataloader
parser.add_argument('--dset', type=str, default='guangzhou', help='dataset name')
parser.add_argument('--subsample', type=bool, default=False, help='Whether to subsample the dataset for quick test')
parser.add_argument('--t_resolution', type=str, default='10T', help='the time resolution for resampling')
parser.add_argument('--patch_len', type=int, default=6, help='patch length')
parser.add_argument('--stride', type=int, default=6, help='stride between patch')
parser.add_argument('--num_patch', type=int, default=36, help='number of patches for one type of flow')
parser.add_argument('--target_len', type=int, default=72, help='forecast horizon for supervised learning')
parser.add_argument('--train_r', type=float, default=0.8, help='the ratio of training data')
parser.add_argument('--val_r', type=float, default=0.1, help='the ratio of validation data')
parser.add_argument('--test_r', type=float, default=0.1, help='the ratio of test data')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--flow_diff_r', type=float, default=0.4, help='the maximum possible ratio of in-out flow '
                                                                   'difference of a day, used to remove outliers')

# RevIN
parser.add_argument('--revin', type=int, default=1, help='whether use reversible instance normalization')

# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
parser.add_argument('--pre_norm', type=bool, default=False, help='whether to apply normalization before attention')
parser.add_argument('--activation', type=str, default='gelu', help='activation function in Transformer')
parser.add_argument('--head_type', type=str, default='prediction', help='one of (prediction, pretrain, MQRnn)')
parser.add_argument('--store_attn', type=bool, default=True, help='whether to store attention weights')
parser.add_argument('--norm', type=str, default='BatchNorm', help='normalization layer, one of (BatchNorm, LayerNorm)')

# positional encoding and feature embedding
parser.add_argument('--pe', type=str, default='zeros', help='type of position encoding (zeros, sincos, or none)')
parser.add_argument('--learn_pe', type=bool, default=True, help='learn position encoding')
parser.add_argument('--flow_eb', type=bool, default=True, help='whether to use flow embedding (inflow and outflow) or not.')
parser.add_argument('--station_eb', type=bool, default=True, help='whether to use station embedding or not.')
parser.add_argument('--weekday_eb', type=bool, default=True, help='whether to use weekday embedding or not.')
parser.add_argument('--time_eb', type=bool, default=True, help='whether to use time embedding or not.')

# Optimization args
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--loss', type=str, default='quantile1', help='loss function, one of (quantile1, quantile3, quantile5)')

args = parser.parse_args([])
print('args:', args)

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

if args.loss == 'quantile1':
    args.out_dim = 1
    args.quantiles = [0.5]
elif args.loss == 'quantile3':
    args.out_dim = 3
    args.quantiles = [0.1, 0.5, 0.9]
elif args.loss == 'quantile5':
    args.out_dim = 5
    args.quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
else:
    raise ValueError('loss function not supported')

#%% Prepare data
args.data_path = '../../../data/GuangzhouMetro//'
args.subsample = False
data_info = data_infos[args.dset]
vars(args).update(data_info)
args.context_len = get_context_len(args.patch_len, args.num_patch, args.stride)

# Set random seed
reset_random_seeds(args.seed)

data = read_data(args)
data = detect_anomaly(data, args)
train_data, val_data, test_data = split_train_val_test(data, args)
train_dataset = MetroDataset(train_data, args)
val_dataset = MetroDataset(val_data, args)
test_dataset = MetroDataset(test_data, args)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#%%
import wandb
def train_MetroTransformer(args, train_loader, val_loader):
    # Determine the number of embeddings
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

    model = MetroTransformer(target_len=args.target_len, patch_len=args.patch_len, num_patch=args.num_patch,
                             num_embeds=num_embeds, n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads,
                             d_ff=args.d_ff, norm=args.norm, dropout=args.dropout,
                             act=args.activation, pre_norm=args.pre_norm, store_attn=args.store_attn,
                             pe=args.pe, learn_pe=args.learn_pe, head_dropout=args.head_dropout,
                             head_type = args.head_type, output_dim=args.out_dim, revin=args.revin)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = lambda x, y: quantile_loss(x, y, args.quantiles)

    # Find the max learning rate
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    max_lr, fig = lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    lr_finder.reset()
    args.max_lr = max_lr
    run = wandb.init(project='MetroTransformer', config=dict(args._get_kwargs()), reinit=True)
    wandb.log({'lr_finder': wandb.Image(fig)})

    args.name = run.name
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.n_epochs)
    step_loss = []
    epoch_loss = []
    lrs = []
    best_val_loss = np.inf
    patience = 5  # Number of epochs with no improvement after which training will be stopped
    epochs_no_improve = 0
    for epoch in range(args.n_epochs):
        model.train()
        for i, (inputs, target) in enumerate(train_loader):
            inputs = [input.to(device) for input in inputs]
            target = target.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(target, output)
            step_loss.append(loss.item())
            lrs.append(scheduler.get_last_lr()[0])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{args.n_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
        epoch_loss.append(np.mean(step_loss[-len(train_loader):]))

        # Calculate validation loss
        model.eval()
        val_loss = []
        for i, (inputs, target) in enumerate(val_loader):
            inputs = [input.to(device) for input in inputs]
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            val_loss.append(loss.item())
        print(f'Epoch [{epoch}/{args.n_epochs}], Val Loss: {np.mean(val_loss):.4f} \t Train Loss: {epoch_loss[-1]:.4f}')
        best_val_loss = min(best_val_loss, np.mean(val_loss))
        wandb.log({'train_loss': epoch_loss[-1], 'val_loss': np.mean(val_loss), 'lr': scheduler.get_last_lr()[0], 'epoch': epoch})

        # Save the current best model
        if np.mean(val_loss) == best_val_loss:
            torch.save(model.state_dict(), f'log//{args.name}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping!')
            break
    # Load the best model
    model.load_state_dict(torch.load(f'log//{args.name}.pth'))

    # Log the training loss
    fig,ax = plt.subplots()
    ax.plot(step_loss)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    wandb.log({"step_loss": wandb.Image(fig)})

    # Log the learning rate
    fig,ax = plt.subplots()
    ax.plot(lrs)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning rate')
    wandb.log({"lr": wandb.Image(fig)})

    wandb.finish()
    return model

train_MetroTransformer(args, train_loader, val_dataset)
#%% Unit test
# a = next(iter(train_loader))
# for i in a[0]:
#     print(i.shape)
#
# a = data.inflow.values.reshape(-1,159, order='F')
# plt.matshow(a)
# # Test if MetroDataset exclude nan values
# train_dataset = MetroDataset(train_data, args)
# for i in range(len(train_dataset)):
#     inputs, _ = train_dataset[i]
#     inflow = inputs[0]
#     station = inputs[2]
#     assert torch.isnan(inflow).sum() == 0, f'nan values in the dataset at index {i}'
#     # If stations are all the same
#     assert torch.all(station == station[0]), f'station values are not the same at index {i}'
#
# # Test if all infeasible index in MetroDataset are infeasible
# for index in train_dataset.if_index:
#     inflow = train_dataset.data.loc[index:index+train_dataset.sample_len, 'inflow'].values
#     station = train_dataset.data.loc[index:index+train_dataset.sample_len, 'station'].values
#     assert (np.isnan(inflow).sum() > 0) or ~np.all(station == station[0]), f'feasible index in infeasible index {index}'
#
# train_dataset.if_index.shape
# train_dataset.f_index.shape
# #
# import wandb
# api = wandb.Api()

# run = api.run("deepzhanhong/MetroTransformer/vg1v6iyw")
# run.config["key"] = updated_value
# run.update()
args = get_wandb_args('deepzhanhong/MetroTransformer/iz8g92dv')
model = MetroTransformer(**args.__dict__)
model.load_state_dict(torch.load(f'log//{args.name}.pth', map_location=torch.device('cpu')))