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

args = get_wandb_args('deepzhanhong/MetroTransformer/ofsy0qvk')
model = MetroTransformer(**args.__dict__)
model.load_state_dict(torch.load(f'logs//{args.name}.pth', map_location=torch.device('cpu')))

#%% prepare data
args.data_path = '../data/GuangzhouMetro//'

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
criterion = lambda x, y: quantile_loss(x, y, args.quantiles)

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