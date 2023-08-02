#%%
import time
script_start_time = time.time()

import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.loss import *
from datasets.datautils import *
from models.NLinear import NLinear
from utilities.lr_finder import LRFinder
import argparse

#%% Define the default arguments
parser = argparse.ArgumentParser()
# General
parser.add_argument('--default_float', type=str, default='float32', help='default float type')
parser.add_argument('--default_int', type=str, default='int32', help='default int type')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--max_run_time', type=int, default=10, help='maximum running time in hours')

# Dataset and dataloader
parser.add_argument('--dset', type=str, default='guangzhou', help='dataset name, guangzhou or nyc or seoul')
parser.add_argument('--subsample', type=bool, default=False, help='Whether to subsample the dataset for quick test')
parser.add_argument('--datatype', type=str, default='Linear', help='one of seq, concat, PatchTST, ...')
parser.add_argument('--train_r', type=float, default=0.8, help='the ratio of training data')
parser.add_argument('--val_r', type=float, default=0.1, help='the ratio of validation data')
parser.add_argument('--test_r', type=float, default=0.1, help='the ratio of test data')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--flow_diff_r', type=float, default=0.4, help='the maximum possible ratio of in-out flow '
                                                                   'difference of a day, used to remove outliers')
parser.add_argument('--standardization', type=str, default='zscore', help='one of (iqr, zscore, minmax)')

# Optimization args
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--loss', type=str, default='rmse', help='loss function, one of '
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
args.dset = "guangzhou"
if args.dset == "guangzhou":
    args.data_path = '../../../data/GuangzhouMetro//'
if args.dset == "seoul":
    args.data_path = '../../../data/SeoulMetro//'

args.model = 'Linear'
args.datatype = 'Linear'
args.subsample = False
args.n_epochs = 20
args.loss = 'rmse'
args.max_lr = 0.005
args.standardization = 'zscore'
args.anneal_strategy = 'linear'

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
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

train_data = train_dataset.data.loc[train_dataset.f_index, :]
x_loc = train_data.groupby('station')['inflow'].mean().values
x_scale = train_data.groupby('station')['inflow'].std().values
x_loc = torch.tensor(x_loc, dtype=torch.float32).to(device)
x_scale = torch.tensor(x_scale, dtype=torch.float32).to(device)

#%% Train model
import wandb
import sys
print(sys.executable)
wandb.login(key='cbe60bf4ccd8041b9a7b7f2946a1c63c85a56a69')
def train_model(args, train_loader, val_loader):
    model = NLinear(x_loc, x_scale, args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = get_loss(args)

    if args.subsample:
        mode = 'disabled'
    else:
        mode = 'online'
    run = wandb.init(project='MetroTransformer', config=dict(args._get_kwargs()), reinit=True, mode='online')

    import datetime
    now = str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + str(datetime.datetime.now().hour)
    wandb.run.name = now + f'_{args.dset}_{args.model}'
    args.name = run.name
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.n_epochs, pct_start=1/args.n_epochs,
                                                    anneal_strategy=args.anneal_strategy)
    step_loss = []
    epoch_loss = []
    lrs = []
    best_val_loss = np.inf
    patience = 20  # Number of epochs with no improvement after which training will be stopped
    epochs_no_improve = 0
    for epoch in range(args.n_epochs): # +args.n_finetune):
        if epoch == args.n_epochs:
            # finetune
            model.load_state_dict(torch.load(f'{args.name}.pth'))

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
            if epoch < args.n_epochs:
                scheduler.step()
            if i % 300 == 0:
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
        print(f'Epoch [{epoch}/{args.n_epochs}], Val Loss: {np.mean(val_loss):.4f} \t Train Loss: {epoch_loss[-1]:.4f} '
              f'\t total time: {time.time() - script_start_time:.2f}')
        best_val_loss = min(best_val_loss, np.mean(val_loss))
        wandb.log({'train_loss': epoch_loss[-1], 'val_loss': np.mean(val_loss), 'lr': scheduler.get_last_lr()[0],
                   'epoch': epoch })

        # Save the current best model
        if np.mean(val_loss) == best_val_loss:
            torch.save(model.state_dict(), f'{args.name}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping!')
            break
        if time.time() - script_start_time > args.max_run_time*3600:
            print(f'Time limit {args.max_run_time} hours reached! Stopping training.')
            break
    # Load the best model
    try:
        model.load_state_dict(torch.load(f'{args.name}.pth'))
    except:
        pass

    # Log the training loss
    fig, ax = plt.subplots()
    ax.plot(step_loss)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    wandb.log({"step_loss": wandb.Image(fig)})

    # Log the learning rate
    fig2, ax2 = plt.subplots()
    ax2.plot(lrs)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning rate')
    wandb.log({"step_learning_rate": wandb.Image(fig2)})
    # wandb.finish()
    return model


model = train_model(args, train_loader, val_loader)

#%% Evaluate the model on the test set
def evaluate(model, test_loader, device='cpu'):
    """Evaluate the model on the test set.
    Returns
    -------
    test_mae, test_rmse
    """
    test_mae = 0
    test_mse = 0
    model.eval()
    len_data = len(test_loader.dataset)
    with torch.no_grad():
        for x,y in test_loader:
            len_batch = len(x[0])
            x = [xx.to(device) for xx in x]
            y = y.to(device)
            y_hat = model(x)
            test_mae += mae_loss(y_hat, y).item() * len_batch / len_data
            test_mse += rmse_loss(y_hat, y).item()**2 * len_batch / len_data
        test_rmse = test_mse**0.5
    return test_mae, test_rmse



test_dataset.mode = 'normal'
test_loader = DataLoader(test_dataset)
normal_test_mae, normal_test_rmse = evaluate(model, test_loader, device=device)
wandb.log({'normal_test_mae': normal_test_mae, 'normal_test_rmse': normal_test_rmse})

test_dataset.mode = 'abnormal'
test_loader = DataLoader(test_dataset)
abnormal_test_mae, abnormal_test_rmse = evaluate(model, test_loader, device=device)
wandb.log({'abnormal_test_mae': abnormal_test_mae, 'abnormal_test_rmse': abnormal_test_rmse})

total_test_mae = (normal_test_mae * len(test_dataset.normal_index) +
                  abnormal_test_mae * len(test_dataset.abnormal_index)) \
                 / len(test_dataset.f_index)
total_test_rmse = ((normal_test_rmse**2 * len(test_dataset.normal_index) +
                    abnormal_test_rmse**2 * len(test_dataset.abnormal_index))
                   / len(test_dataset.f_index))**0.5
wandb.log({'total_test_mae': total_test_mae, 'total_test_rmse': total_test_rmse})
wandb.finish()
