#%% Forecast alighting and boarding together but in a concatenated way, without flow type embedding, with ABT_new (faster forecasting)
import time, datetime
script_start_time = time.time()

import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.basics import *
from datasets.datautils import *
from models.ABT_new import ABTransformer
from torch.utils.data import DataLoader, RandomSampler

#%% Define arguments & prepare data
args.dset = "seoul"  # "seoul" or "guangzhou"
vars(args).update(data_infos[args.dset])
args.model = 'ABT'
vars(args).update(model_infos[args.model])
for head_type in ['TruncatedNormal', 'CrossEntropy','MixTruncatedNormal']:
    args.head_type = head_type  # 'RMSE' or 'NB' or CrossEntropy, TruncatedNormal, Normal, MixTruncatedNormal
    vars(args).update(head_infos[args.head_type])
    args.n_epochs = 20
    args.mode = 'online'  # online or disabled
    vars(args).update(basic_infos)
    args.dropout = 0.05
    args.head_dropout = 0.05
    args.attn_dropout = 0.05
    args.weighted_loss = False
    # args.train_method = 'CRPS'  # whether to use CRPS to train the categorical distribution
    args.d_ff = 256
    # args.initial_num_bins = 1024
    args.pe = 'rotary'
    args.patience = 10
    args.forecast_target = 'both'

    data = pd.read_csv(args.data_path+'data.csv', parse_dates=['time'], dtype={'station': args.default_int,
                                                                               'inflow': args.default_float,
                                                                               'outflow': args.default_float,
                                                                               'weekday': args.default_int,
                                                                               'time_in_day': args.default_int,
                                                                               'abnormal_in': 'bool',
                                                                               'abnormal_out':'bool'})
    dataset = MetroDataset_total(data, args)
    train_dataset = dataset.TrainDataset
    val_dataset = dataset.ValDataset
    test_dataset = dataset.TestDataset

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=len(train_dataset)//args.sample_divide)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    args.num_embeds = get_num_embedding(args, train_loader)
    args.num_patch = dataset.num_input_patch
    args.num_target_patch = dataset.num_sample_patch - dataset.num_input_patch
    args.num_bins = dataset.num_bins  # Update the number of bins
    b = train_loader.dataset.raw_data.loc[train_loader.dataset.f_index, ['inflow', 'outflow']].values.max()
    args.b = b

    # %% Get x_loc and x_scale
    x_loc, x_scale = get_loc_scale(train_dataset.data.loc[train_dataset.f_index, :], args.standardization)
    x_loc = torch.tensor(x_loc, dtype=torch.float32).to(device)
    x_scale = torch.tensor(x_scale, dtype=torch.float32).to(device)
    print('args:', args)
    args.bin_edges = dataset.bin_edges  # Update the bin edges
    if args.weighted_loss:
        args.bin_weights = dataset.bin_weights  # Update the bin weights
    else:
        args.bin_weights = None
    attn_mask = get_attn_mask(args.num_patch, args.num_target_patch - 1, device=device, type=args.attn_mask_type)


    # %% Train model
    wandb.login()
    reset_random_seeds(args.seed)
    model = ABTransformer(x_loc=x_loc, x_scale=x_scale, attn_mask=attn_mask, **vars(args))
    model = train_model(args, train_loader, val_loader, test_loader, model, device=device)

#%%
