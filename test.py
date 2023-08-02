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
import wandb

#%% Test the performance of old methods
args = get_wandb_args('deepzhanhong/MetroTransformer/utadtdug')
args.data_path = '../data/GuangzhouMetro//'
device = 'cpu'
data = read_data(args)
data = detect_anomaly(data, args)
train_data, val_data, test_data = split_train_val_test(data, args)
train_dataset = MetroDataset(train_data, 'target', args)
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

model0 = MetroTransformer(x_loc=x_loc, x_scale=x_scale, **args.__dict__)
model0.load_state_dict(torch.load(f'exps//ABtransformer//log//{args.name}.pth', map_location=torch.device('cpu')))

dataset = MetroDataset_total(data, args, datatype='seq')
train_dataset = dataset.TrainDataset
val_dataset = dataset.ValDataset
test_dataset = dataset.TestDataset
train_dataset.mask_method = args.data_mask_method; train_dataset.mask_ratio = args.data_mask_ratio
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


def evaluate_model(model, test_dataset):
    test_dataset.mode = 'normal'
    test_loader = DataLoader(test_dataset)
    normal_test_mae, normal_test_rmse = evaluate(model, test_loader, device=device)
    print(f'Normal test MAE: {normal_test_mae:.2f}, Normal test RMSE: {normal_test_rmse:.2f}')
    # wandb.log({'normal_test_mae': normal_test_mae, 'normal_test_rmse': normal_test_rmse})

    test_dataset.mode = 'abnormal'
    test_loader = DataLoader(test_dataset)
    abnormal_test_mae, abnormal_test_rmse = evaluate(model, test_loader, device=device)
    print(f'Abnormal test MAE: {abnormal_test_mae:.2f}, Abnormal test RMSE: {abnormal_test_rmse:.2f}')
    # wandb.log({'abnormal_test_mae': abnormal_test_mae, 'abnormal_test_rmse': abnormal_test_rmse})

    total_test_mae = (normal_test_mae * len(test_dataset.normal_index) +
                      abnormal_test_mae * len(test_dataset.abnormal_index)) \
                     / len(test_dataset.f_index)
    total_test_rmse = ((normal_test_rmse**2 * len(test_dataset.normal_index) +
                        abnormal_test_rmse**2 * len(test_dataset.abnormal_index))
                       / len(test_dataset.f_index))**0.5
    print(f'Total test MAE: {total_test_mae:.2f}, Total test RMSE: {total_test_rmse:.2f}')

evaluate_model(model0, val_dataset)

#%% Test the performance of new methods
args = get_wandb_args('deepzhanhong/MetroTransformer/uxffd5hv')
args.data_path = '../data/GuangzhouMetro//'
device = 'cpu'
data = read_data(args)
data = detect_anomaly(data, args)
dataset2 = MetroDataset_total(data, args, datatype='seq')
train_dataset2 = dataset2.TrainDataset
val_dataset = dataset2.ValDataset
test_dataset = dataset2.TestDataset
train_dataset2.mask_method = args.data_mask_method; train_dataset2.mask_ratio = args.data_mask_ratio
train_loader = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
train_data2 = train_dataset2.data.loc[train_dataset2.f_index, :]
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

model1 = MetroTransformer(x_loc=x_loc, x_scale=x_scale, **args.__dict__)
model1.load_state_dict(torch.load(f'exps//ABtransformer//log//{args.name}.pth', map_location=torch.device('cpu')))
evaluate_model(model1, test_dataset)

#%% Test the prediction results on the training set
def exam_result(t, s, model, dataset, args):
    t = pd.Timestamp(t)
    (x, y), data_piece = dataset.get_data_from_ts(t, s)

    x = [xx.unsqueeze(0) for xx in x]
    model.eval()
    y_hat = model(x)

    fig, ax = plt.subplots()
    ax.plot(data_piece['time'].values, data_piece['inflow'].values, label='Real boarding', color='C0')
    ax.plot(data_piece['time'].values[:args.context_len],
             data_piece['outflow'].values[:args.context_len],
             label='Real alighting', color='C1')
    ax.plot(data_piece['time'].values[args.context_len:], y_hat.detach().numpy(), label='Prediction', color='C2')
    ax.set_xlabel('Time')
    # rotate xticks labels
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.legend()
    plt.show()

exam_result(t='2017-07-28 18:00:00', s = 110, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-07-29 20:30:00', s = 33, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-07-27 18:00:00', s = 33, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-09-16 18:00:00', s = 33, model=model, dataset=val_dataset, args=args)
exam_result(t='2017-09-13 18:00:00', s = 31, model=model, dataset=val_dataset, args=args)
# Station 44
exam_result(t='2017-07-13 18:00:00', s = 44, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-07-20 18:00:00', s = 44, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-07-27 18:00:00', s = 44, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-07-29 18:00:00', s = 44, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-07-30 18:00:00', s = 44, model=model, dataset=train_dataset, args=args)
exam_result(t='2017-08-06 18:00:00', s = 44, model=model, dataset=train_dataset, args=args)


exam_result(t='2017-09-13 18:00:00', s = 44, model=model, dataset=val_dataset, args=args)
exam_result(t='2017-09-14 18:00:00', s = 44, model=model, dataset=val_dataset, args=args)
exam_result(t='2017-09-13 18:00:00', s = 44, model=model, dataset=val_dataset, args=args)
#%% Test the prediction results on the validation set
(x, y), data_piece = train_dataset.get_data_from_ts(pd.Timestamp('2017-07-28 18:00:00'), 110)
a = x[0].detach().numpy()
x = [xx.unsqueeze(0) for xx in x]
model.eval()
y_hat = model(x)

(x, y) = next(train_loader.__iter__())
model.eval()
y_hat = model(x)
y_hat.shape
y.shape
quantile_loss(y_hat, y, args.quantiles)
len(np.setdiff1d(train_data.index.values, train_data2.index.values))
train_data = train_data.sort_values(['station', 'time'])
train_data2 = train_data2.sort_values(['station', 'time'])
np.isnan(train_data2.inflow.values).sum()

x, y = next(train_loader.__iter__())
y.shape
for xx in x:
    print(xx.shape)

a = x[1].detach().numpy()
b = x[-1].detach().numpy()
# data, flow_type, station, weekday, time_in_day, fcst_loc