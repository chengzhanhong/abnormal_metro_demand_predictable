#%%
import time, datetime
script_start_time = time.time()

import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.basics import *
from datasets.datautils import *
from models.basics import *
from models.NLinear import NLinear
from torch.utils.data import DataLoader, RandomSampler

#%% Define arguments & prepare data
for dset in ['guangzhou', 'seoul']:
    for forecast_target in ['outflow']:
        args.dset = dset  # "seoul" or "guangzhou
        vars(args).update(data_infos[args.dset])
        args.model = 'Nlinear'
        vars(args).update(model_infos[args.model])
        args.input_len = args.input_len//2
        args.target_len = args.target_len//2
        args.head_type = 'RMSE'
        vars(args).update(head_infos[args.head_type])
        args.n_epochs = 20
        args.mode = 'online'  # online or disabled

        args.dropout = 0.05
        args.max_lr = 0.001
        args.patience = 5
        args.anneal_strategy = 'linear'
        args.batch_size = 128
        args.patch_len = 1
        args.flow_eb = False
        args.station_eb = True
        args.weekday_eb = True
        args.time_eb = True
        args.seed = 0
        args.max_run_time = 10  # in hours
        args.div_factor = 1e4  # initial warmup learning rate = max_lr / div_factor
        args.final_div_factor = 1  # final learning rate = initial_lr / final_div_factor
        args.input_emb_size = 8
        args.max_leap_rate = 0.1
        args.initial_num_bins = 1024
        args.forecast_target = forecast_target
        args.d_model = 64

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


        # %% Get x_loc and x_scale
        train_data = train_dataset.data.loc[train_dataset.f_index, :]
        x_loc, x_scale = get_loc_scale(train_data, args.standardization)
        x_loc = torch.tensor(x_loc, dtype=torch.float32).to(device)
        x_scale = torch.tensor(x_scale, dtype=torch.float32).to(device)
        print('args:', args)

        #%% Train model
        wandb.login(key='cbe60bf4ccd8041b9a7b7f2946a1c63c85a56a69')
        reset_random_seeds(args.seed)
        model = NLinear(x_loc=x_loc, x_scale=x_scale, **vars(args))
        model = train_model(args, train_loader, val_loader, test_loader, model, device=device)
