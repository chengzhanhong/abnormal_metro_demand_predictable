# %%
import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.basics import *
from datasets.datautils import *
from models.ABT_concat import ABTransformer_concat
from models.ABT import ABTransformer
from models.DeepAR import DeepAR

# args = get_wandb_args('deepzhanhong/ABTransformer/0ylvdm3d')  # ABT_concat, https://wandb.ai/deepzhanhong/ABTransformer/runs/0ylvdm3d/overview?nw=nwuserzhanhong
# args = get_wandb_args('deepzhanhong/ABTransformer/j00e1bko')  # ABT_concat, https://wandb.ai/deepzhanhong/ABTransformer/runs/j00e1bko/overview?nw=nwuserzhanhong
# args = get_wandb_args('deepzhanhong/ABTransformer/mugaugw1')  # ABT https://wandb.ai/deepzhanhong/ABTransformer/runs/mugaugw1/overview?nw=nwuserzhanhong
# args = get_wandb_args('deepzhanhong/ABTransformer/03vrv246')  # DeepAR, https://wandb.ai/deepzhanhong/ABTransformer/runs/03vrv246/overview?nw=nwuserzhanhong
args = get_wandb_args('deepzhanhong/ABTransformer/0ylvdm3d')
# strip the leading ../ from the path
args.data_path = args.data_path[3:]
args.attn_mask_type = 'time'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)
else:
    print('No GPU available, using the CPU instead.')

data = pd.read_csv(args.data_path+'data.csv', parse_dates=['time'], dtype={'station': args.default_int,
                                                                           'inflow': args.default_float,
                                                                           'outflow': args.default_float,
                                                                           'weekday': args.default_int,
                                                                           'time_in_day': args.default_int,
                                                                           'abnormal_in': 'bool',
                                                                           'abnormal_out':'bool'})

dataset = MetroDataset_total(data, args, datatype=args.datatype)
train_dataset = dataset.TrainDataset
val_dataset = dataset.ValDataset
test_dataset = dataset.TestDataset
test_dataset.return_abnormal = True

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

x_loc, x_scale = get_loc_scale(train_dataset.data.loc[train_dataset.f_index, :], args.standardization)
x_loc = torch.tensor(x_loc, dtype=torch.float32).to(device)
x_scale = torch.tensor(x_scale, dtype=torch.float32).to(device)
attn_mask = get_attn_mask(args.num_patch, args.num_target_patch-1, device=device,
                          type=args.attn_mask_type)
models = {'ABT': ABTransformer, 'ABT_concat': ABTransformer_concat, 'DeepAR': DeepAR}
paths = {'ABT': 'ABTransformer\\log\\', 'ABT_concat': 'ABTransformer\\log\\', 'DeepAR': 'DeepAR\\log\\'}
model = models[args.model](x_loc=x_loc, x_scale=x_scale, attn_mask=attn_mask, **vars(args))
model.to(device)
model.load_state_dict(torch.load(f'{paths[args.model]}{args.name}.pth'))
#%%
model.eval()
ys = []
abnormals = []
results = []
for x, y, abnormal in test_loader:
    x = [xx.to(device) for xx in x]
    y_hat = model.forecast(x)
    results.append(y_hat)
    ys.append(y[:, -args.num_target_patch:, :])  # (batch_size, num_target_patch, patch_len*2)
    abnormals.append(abnormal[:, -args.num_target_patch:, :])

results = torch.cat(results, dim=0).cpu().detach().numpy()  # (num_samples, num_target_patch, patch_len*2)
ys = torch.cat(ys, dim=0).cpu().detach().numpy()
abnormals = torch.cat(abnormals, dim=0).cpu().detach().numpy()
inflows = results[:, :, :args.patch_len]
outflows = results[:, :, args.patch_len:]
y_inflows = ys[:, :, :args.patch_len]
y_outflows = ys[:, :, args.patch_len:]
ab_inflows = abnormals[:, :, :args.patch_len]
ab_outflows = abnormals[:, :, args.patch_len:]
rmse = lambda x, y: np.sqrt(np.mean((x-y)**2))
mae = lambda x, y: np.mean(np.abs(x-y))

# Total
total_rmse = rmse(results, ys)
total_mae = mae(results, ys)
normal_rmse = rmse(results[~abnormals], ys[~abnormals])
normal_mae = mae(results[~abnormals], ys[~abnormals])
abnormal_rmse = rmse(results[abnormals], ys[abnormals])
abnormal_mae = mae(results[abnormals], ys[abnormals])

# Inflow
total_rmse_inflow = rmse(inflows, y_inflows)
total_mae_inflow = mae(inflows, y_inflows)
normal_rmse_inflow = rmse(inflows[~ab_inflows], y_inflows[~ab_inflows])
normal_mae_inflow = mae(inflows[~ab_inflows], y_inflows[~ab_inflows])
abnormal_rmse_inflow = rmse(inflows[ab_inflows], y_inflows[ab_inflows])
abnormal_mae_inflow = mae(inflows[ab_inflows], y_inflows[ab_inflows])

# Outflow
total_rmse_outflow = rmse(outflows, y_outflows)
total_mae_outflow = mae(outflows, y_outflows)
normal_rmse_outflow = rmse(outflows[~ab_outflows], y_outflows[~ab_outflows])
normal_mae_outflow = mae(outflows[~ab_outflows], y_outflows[~ab_outflows])
abnormal_rmse_outflow = rmse(outflows[ab_outflows], y_outflows[ab_outflows])
abnormal_mae_outflow = mae(outflows[ab_outflows], y_outflows[ab_outflows])

print(f'Total test RMSE: {total_rmse:.4f}, Total test MAE: {total_mae:.4f}')
print(f'Normal test RMSE: {normal_rmse:.4f}, Normal test MAE: {normal_mae:.4f}')
print(f'Abnormal test RMSE: {abnormal_rmse:.4f}, Abnormal test MAE: {abnormal_mae:.4f}')
print(f'Total test RMSE inflow: {total_rmse_inflow:.4f}, Total test MAE inflow: {total_mae_inflow:.4f}')
print(f'Normal test RMSE inflow: {normal_rmse_inflow:.4f}, Normal test MAE inflow: {normal_mae_inflow:.4f}')
print(f'Abnormal test RMSE inflow: {abnormal_rmse_inflow:.4f}, Abnormal test MAE inflow: {abnormal_mae_inflow:.4f}')
print(f'Total test RMSE outflow: {total_rmse_outflow:.4f}, Total test MAE outflow: {total_mae_outflow:.4f}')
print(f'Normal test RMSE outflow: {normal_rmse_outflow:.4f}, Normal test MAE outflow: {normal_mae_outflow:.4f}')
print(f'Abnormal test RMSE outflow: {abnormal_rmse_outflow:.4f}, Abnormal test MAE outflow: {abnormal_mae_outflow:.4f}')

rmses = {
    'normal_rmse_inflow': normal_rmse_inflow,
    'abnormal_rmse_inflow': abnormal_rmse_inflow,
    'total_rmse_inflow': total_rmse_inflow,
    'normal_rmse_outflow': normal_rmse_outflow,
    'abnormal_rmse_outflow': abnormal_rmse_outflow,
    'total_rmse_outflow': total_rmse_outflow,
    'normal_rmse': normal_rmse,
    'abnormal_rmse': abnormal_rmse,
    'total_rmse': total_rmse
}
rmses = pd.DataFrame(rmses, index=[0])
rmses
maes = {
    'normal_mae': normal_mae,
    'abnormal_mae': abnormal_mae,
    'total_mae': total_mae,
    'normal_mae_inflow': normal_mae_inflow,
    'abnormal_mae_inflow': abnormal_mae_inflow,
    'total_mae_inflow': total_mae_inflow,
    'normal_mae_outflow': normal_mae_outflow,
    'abnormal_mae_outflow': abnormal_mae_outflow,
    'total_mae_outflow': total_mae_outflow,
}
maes = pd.DataFrame(maes, index=[0])
maes

print('Step by step RMSE:')
for p in range(args.num_target_patch):
    for i in range(args.patch_len):
        y_step = ys[:, p, i::args.patch_len]
        result_step = results[:, p, i::args.patch_len]
        print(f'{rmse(result_step, y_step):.4f}\t', end='')

#%% [2, patch_len, num_patch, num_target_patch]

t = '2023-04-01 14:00:00'
s = 27
t1 = pd.Timestamp(t)
# (x, y), data_piece = test_dataset.get_data_from_ts(t1, s)
(x, y, _), data_piece = train_dataset.get_data_from_ts(t1, s)
x = [xx.unsqueeze(0) for xx in x]
x = [xx.to(device) for xx in x]
y_hat = model.forecast(x)   # (1, num_fcst, patch_len, output_dim)
y_hat = y_hat.squeeze()

len_y = len(y)
len_y_hat = len(y_hat)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_piece.time, data_piece.inflow, label='ground truth')
ax.plot(data_piece.time[-len_y_hat:], y_hat.squeeze().numpy(), label='prediction')


# #%%
# total_rmse, total_mae, normal_rmse, normal_mae, abnormal_rmse, abnormal_mae = evaluate(model, test_loader, device=device)
# print(f'Total test RMSE: {total_rmse:.4f}, Total test MAE: {total_mae:.4f}')
# print(f'Normal test RMSE: {normal_rmse:.4f}, Normal test MAE: {normal_mae:.4f}')
# print(f'Abnormal test RMSE: {abnormal_rmse:.4f}, Abnormal test MAE: {abnormal_mae:.4f}')
































