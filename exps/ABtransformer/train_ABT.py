#%% Forecast alighting and boarding together but in a concatenated way
import time, datetime
script_start_time = time.time()

import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.basics import *
from utilities.loss import *
from datasets.datautils import *
from models.ABT import ABTransformer

#%% Define arguments & prepare data
args.dset = "guangzhou"
vars(args).update(data_infos[args.dset])
args.model = 'ABT'
vars(args).update(model_infos[args.model])
parser = argparse.ArgumentParser()

args.datatype = 'ABT'
args.loss = 'rmse'
args.mode = 'online'  # online or disabled
args.dropout = 0.05
args.attn_dropout = 0

args.n_epochs = 20
args.max_lr = 0.001
args.patience = 5
args.standardization = 'zscore'  # 'zscore' or 'minmax', 'none', 'meanscale'
args.anneal_strategy = 'linear'
args.batch_size = 128
args.d_model = 128
args.n_heads = 8
args.n_layers = 3
args.attn_mask_type = 'time'
args.pre_norm = True
args.pe = 'zeros'  # intial values of positional encoding
args.learn_pe = True  # learn positional encoding
args.anneal_strategy = 'linear'
args.div_factor = 1e4  # initial warmup learning rate = max_lr / div_factor
args.final_div_factor = 1  # final learning rate = initial_lr / final_div_factor

data = pd.read_csv(args.data_path+'data.csv', parse_dates=['time'], dtype={'station': args.default_int,
                                                                           'inflow': args.default_float,
                                                                           'outflow': args.default_float,
                                                                           'weekday': args.default_int,
                                                                           'time_in_day': args.default_int,
                                                                           'abnormal_in': 'bool',
                                                                           'abnormal_out':'bool'})
data.inflow.argmax()
dataset = MetroDataset_total(data, args, datatype=args.datatype)
train_dataset = dataset.TrainDataset
val_dataset = dataset.ValDataset
test_dataset = dataset.TestDataset

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
args.num_embeds = get_num_embedding(args, train_loader)
args.num_patch = dataset.num_input_patch
args.num_target_patch = dataset.num_sample_patch - dataset.num_input_patch

# %% Get x_loc and x_scale
x_loc, x_scale = get_loc_scale(train_dataset.data.loc[train_dataset.f_index, :], args.standardization)
x_loc = torch.tensor(x_loc, dtype=torch.float32).to(device)
x_scale = torch.tensor(x_scale, dtype=torch.float32).to(device)
print('args:', args)

# %% Inspect the data
x, y = next(iter(train_loader))
# val_data = val_dataset.data.loc[val_dataset.f_index, :]
# test_data = test_dataset.data.loc[test_dataset.f_index, :]
attn_mask = get_attn_mask(args.num_patch, args.num_target_patch - 1, device=device, type=args.attn_mask_type)

# %% Train model
# Main experiments
import wandb
wandb.login(key='cbe60bf4ccd8041b9a7b7f2946a1c63c85a56a69')

def train_model(args, train_loader, val_loader):
    attn_mask = get_attn_mask(args.num_patch, args.num_target_patch - 1, device=device, type=args.attn_mask_type)
    model = ABTransformer(x_loc=x_loc, x_scale=x_scale, attn_mask=attn_mask, **vars(args))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = get_loss(args)
    run = wandb.init(project='ABTransformer', config=dict(args._get_kwargs()), reinit=True, mode=args.mode)

    import datetime
    now = f'{datetime.datetime.now().month:02d}_{datetime.datetime.now().day:02d}_{datetime.datetime.now().hour:02d}'
    wandb.run.name = now + f'_{args.dset}_{args.model}'
    args.name = run.name
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.n_epochs, pct_start=1/args.n_epochs,
                                                    div_factor=args.div_factor, final_div_factor=args.final_div_factor,
                                                    anneal_strategy=args.anneal_strategy)
    step_loss = []
    epoch_loss = []
    lrs = []
    best_val_loss = np.inf
    patience = args.patience  # Number of epochs with no improvement after which training will be stopped
    epochs_no_improve = 0
    for epoch in range(args.n_epochs):
        model.train()
        for i, (inputs, target) in enumerate(train_loader):
            inputs = [input.to(device) for input in inputs]
            target = target.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            step_loss.append(loss.item())
            lrs.append(scheduler.get_last_lr()[0])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 300 == 0:
                print(f'Epoch [{epoch}/{args.n_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}'
                      f'\t total time: {time.time() - script_start_time:.2f}')

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

        print(f'Epoch [{epoch}/{args.n_epochs}], Val Loss: {np.mean(val_loss):.4f} \t Train Loss: {epoch_loss[-1]:.4f} '
              f'\t total time: {time.time() - script_start_time:.2f}')
        best_val_loss = min(best_val_loss, np.mean(val_loss))
        wandb.log({'train_loss': epoch_loss[-1], 'val_loss': np.mean(val_loss),
                   'epoch': epoch})

        # Save the current best model
        if np.mean(val_loss) == best_val_loss:
            torch.save(model.state_dict(), f'log\\{args.name}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping!')
            break
        if time.time() - script_start_time > args.max_run_time * 3600:
            print(f'Time limit {args.max_run_time} hours reached! Stopping training.')
            break

    # Load the best model
    if np.mean(val_loss) != best_val_loss:
        model.load_state_dict(torch.load(f'log\\{args.name}.pth'))

    # Test the model
    model.eval()
    test_loss = []
    for x, y in test_loader:
        x = [xx.to(device) for xx in x]
        y = y.to(device)
        y_hat = model.forecast(x)
        loss = criterion(y_hat, y[:, -args.num_target_patch:, :])
        test_loss.append(loss.item())
    print(f'Test Loss: {np.mean(test_loss):.4f}')
    wandb.log({'test_loss': np.mean(test_loss)})

    wandb.finish()

    return model

reset_random_seeds(args.seed)
model = train_model(args, train_loader, val_loader)
