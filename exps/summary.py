# %%
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.basics import *
from datasets.datautils import *
# from models.ABT_concat_new import ABTransformer_concat
from models.ABT_concat import ABTransformer_concat
from models.ABT_new import ABTransformer
from models.DeepAR import DeepAR
from models.NLinear import NLinear
from models.basics import *
import pickle
import properscoring as pscore
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)
else:
    print('No GPU available, using the CPU instead.')


#%%
def evaluate(wab_id, model_description, n=50):
    args = get_wandb_args(f'deepzhanhong/ABTransformer/{wab_id}')
    args.initial_num_bins = 1024
    args.data_path = args.data_path[3:]
    args.attn_mask_type = 'time'
    if 'forecast_target' not in args:
        args.forecast_target = 'both'
    models = {'ABT': ABTransformer, 'ABT_concat': ABTransformer_concat, 'DeepAR': DeepAR, 'ABT_new': ABTransformer,
              'ABT2': ABTransformer, 'Nlinear': NLinear}
    paths = {'ABT': 'ABTransformer\\log\\', 'ABT_concat': 'ABTransformer\\log\\',
             'DeepAR': 'DeepAR\\log\\', 'ABT_new': 'ABTransformer\\log\\', 'ABT2': 'ABTransformer\\log\\',
             'Nlinear': 'NLinear\\log\\'}
    if 'sample_divide' in args and args.model != 'Nlinear':
        args.num_patch = args.num_patch//2
        args.num_target_patch = args.num_target_patch//2
        args.input_len = args.input_len//2
        args.target_len = args.target_len//2
    args.sample_interval = 1 if args.dset == 'seoul' else 2
    complete = False

    # if logs/args.name_results.pkl exists, load the results and return
    if os.path.exists(f'logs//{args.name}_results_{n}.pkl'):
        with open(f'logs//{args.name}_results_{n}.pkl', 'rb') as f:
            results = pickle.load(f)
        try:
            rmses_dict = results['rmses_dict']
            maes_dict = results['maes_dict']
            crps_dict = results['crps_dict']
            wmapes_dict = results['wmapes']
            print(f'Results loaded from logs//{args.name}_results_{n}.pkl')
            complete = True
        except:
            print(f'Some results are missing in logs//{args.name}_results_{n}.pkl, re-evaluate.')
            complete = False

    if not complete:
        data = pd.read_csv(args.data_path+'data.csv', parse_dates=['time'], dtype={'station': args.default_int,
                                                                                   'inflow': args.default_float,
                                                                                   'outflow': args.default_float,
                                                                                   'weekday': args.default_int,
                                                                                   'time_in_day': args.default_int,
                                                                                   'abnormal_in': 'bool',
                                                                                   'abnormal_out':'bool'})

        dataset = MetroDataset_total(data, args)
        train_dataset = dataset.TrainDataset
        test_dataset = dataset.TestDataset
        test_dataset.test_mode = True

        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        args.bin_edges = dataset.bin_edges  # Update the bin edges

        x_loc, x_scale = get_loc_scale(train_dataset.data.loc[train_dataset.f_index, :], args.standardization)
        x_loc = torch.tensor(x_loc, dtype=torch.float32).to(device)
        x_scale = torch.tensor(x_scale, dtype=torch.float32).to(device)
        attn_mask = get_attn_mask(args.num_patch, args.num_target_patch-1, device=device,
                                  type=args.attn_mask_type)

        model = models[args.model](x_loc=x_loc, x_scale=x_scale, attn_mask=attn_mask, **vars(args))
        model.to(device)
        if 'pe' in args:
            if args.pe == 'rotary':
                model.backbone.W_pos = None
        model.load_state_dict(torch.load(f'{paths[args.model]}{args.name}.pth'), strict=False)

        model.eval()
        test_loader.dataset.test_mode = True
        num_batches = len(test_loader)
        ys = []
        abnormals = []
        results = []
        print(f'Start the forecasting of {model_description}:\n')
        time0 = time.time()
        with torch.no_grad():
            for i, (x, y, abnormal) in enumerate(test_loader):
                x = [xx.to(device) for xx in x]
                ys.append(y[:, -args.num_target_patch:, :])  # (batch_size, num_target_patch, patch_len*2)
                if args.head_type != 'RMSE':
                    y_hat = model.forecast_samples(x, n=n)   # (1, num_fcst, patch_len, output_dim)
                else:
                    y_hat = model.forecast(x)
                abnormals.append(abnormal[:, -args.num_target_patch:, :])
                results.append(y_hat)
                if i % 100 == 0:
                    print(f'{i}/{num_batches} batches done, time elapsed: {time.time()-time0:.2f}s')

        ys = torch.cat(ys, dim=0).cpu().detach().numpy()
        abnormals = torch.cat(abnormals, dim=0).cpu().detach().numpy()
        if args.head_type != 'RMSE':
            results = torch.cat(results, dim=1).cpu().detach().numpy()  # (num_samples, num_target_patch, patch_len*2)
            mean_forecast = results.mean(axis=0)
        else:
            mean_forecast = torch.cat(results, dim=0).cpu().detach().numpy()

        inflow_slice = slice(0, args.patch_len)
        outflow_slice = slice(0, args.patch_len) if args.forecast_target in {'outflow', 'inflow'} else slice(args.patch_len, None)
        inflows = mean_forecast[:, :, inflow_slice]
        outflows = mean_forecast[:, :, outflow_slice]
        y_inflows = ys[:, :, inflow_slice]
        y_outflows = ys[:, :, outflow_slice]
        ab_inflows = abnormals[:, :, inflow_slice]
        ab_outflows = abnormals[:, :, outflow_slice]
        rmse = lambda x, y: np.sqrt(np.mean((x-y)**2))
        mae = lambda x, y: np.mean(np.abs(x-y))
        wmape = lambda x, y: np.sum(np.abs(x-y))/np.sum(y)

        # Total
        total_rmse = rmse(mean_forecast, ys)
        total_mae = mae(mean_forecast, ys)
        total_wmape = wmape(mean_forecast, ys)
        normal_rmse = rmse(mean_forecast[~abnormals], ys[~abnormals])
        normal_mae = mae(mean_forecast[~abnormals], ys[~abnormals])
        normal_wmape = wmape(mean_forecast[~abnormals], ys[~abnormals])
        abnormal_rmse = rmse(mean_forecast[abnormals], ys[abnormals])
        abnormal_mae = mae(mean_forecast[abnormals], ys[abnormals])
        abnormal_wmape = wmape(mean_forecast[abnormals], ys[abnormals])

        # # Inflow
        total_rmse_inflow = rmse(inflows, y_inflows)
        total_mae_inflow = mae(inflows, y_inflows)
        total_wmape_inflow = wmape(inflows, y_inflows)
        normal_rmse_inflow = rmse(inflows[~ab_inflows], y_inflows[~ab_inflows])
        normal_mae_inflow = mae(inflows[~ab_inflows], y_inflows[~ab_inflows])
        normal_wmape_inflow = wmape(inflows[~ab_inflows], y_inflows[~ab_inflows])
        abnormal_rmse_inflow = rmse(inflows[ab_inflows], y_inflows[ab_inflows])
        abnormal_mae_inflow = mae(inflows[ab_inflows], y_inflows[ab_inflows])
        abnormal_wmape_inflow = wmape(inflows[ab_inflows], y_inflows[ab_inflows])

        # Outflow
        total_rmse_outflow = rmse(outflows, y_outflows)
        total_mae_outflow = mae(outflows, y_outflows)
        total_wmape_outflow = wmape(outflows, y_outflows)
        normal_rmse_outflow = rmse(outflows[~ab_outflows], y_outflows[~ab_outflows])
        normal_mae_outflow = mae(outflows[~ab_outflows], y_outflows[~ab_outflows])
        normal_wmape_outflow = wmape(outflows[~ab_outflows], y_outflows[~ab_outflows])
        abnormal_rmse_outflow = rmse(outflows[ab_outflows], y_outflows[ab_outflows])
        abnormal_mae_outflow = mae(outflows[ab_outflows], y_outflows[ab_outflows])
        abnormal_wmape_outflow = wmape(outflows[ab_outflows], y_outflows[ab_outflows])

        rmses_dict = {
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


        maes_dict = {
            'normal_mae_inflow': normal_mae_inflow,
            'abnormal_mae_inflow': abnormal_mae_inflow,
            'total_mae_inflow': total_mae_inflow,
            'normal_mae_outflow': normal_mae_outflow,
            'abnormal_mae_outflow': abnormal_mae_outflow,
            'total_mae_outflow': total_mae_outflow,
            'normal_mae': normal_mae,
            'abnormal_mae': abnormal_mae,
            'total_mae': total_mae,
        }

        wmapes_dict = {
            'normal_wmape_inflow': normal_wmape_inflow,
            'abnormal_wmape_inflow': abnormal_wmape_inflow,
            'total_wmape_inflow': total_wmape_inflow,
            'normal_wmape_outflow': normal_wmape_outflow,
            'abnormal_wmape_outflow': abnormal_wmape_outflow,
            'total_wmape_outflow': total_wmape_outflow,
            'normal_wmape': normal_wmape,
            'abnormal_wmape': abnormal_wmape,
            'total_wmape': total_wmape,
        }

        if args.head_type != 'RMSE':
            # Compute the crps
            crps_result = np.zeros(ys.shape)
            results1 = results.transpose(1, 2, 3, 0)
            for i in range(ys.shape[0]):
                crps_result[i, :, :] = pscore.crps_ensemble(ys[i, :, :], results1[i])

            total_crps = crps_result.mean()
            normal_crps = crps_result[~abnormals].mean()
            abnormal_crps = crps_result[abnormals].mean()

            crps_inflows = crps_result[:, :, :args.patch_len]
            total_crps_inflow = crps_inflows.mean()
            normal_crps_inflow = crps_inflows[~ab_inflows].mean()
            abnormal_crps_inflow = crps_inflows[ab_inflows].mean()

            crps_outflows = crps_result[:, :, args.patch_len:]
            total_crps_outflow = crps_outflows.mean()
            normal_crps_outflow = crps_outflows[~ab_outflows].mean()
            abnormal_crps_outflow = crps_outflows[ab_outflows].mean()
            crps_dict = {
                'normal_crps_inflow': normal_crps_inflow,
                'abnormal_crps_inflow': abnormal_crps_inflow,
                'total_crps_inflow': total_crps_inflow,
                'normal_crps_outflow': normal_crps_outflow,
                'abnormal_crps_outflow': abnormal_crps_outflow,
                'total_crps_outflow': total_crps_outflow,
                'normal_crps': normal_crps,
                'abnormal_crps': abnormal_crps,
                'total_crps': total_crps
            }
        else:
            crps_dict = {}


    print(f'The result of {model_description}:\n')
    print(rmses_dict)
    print(maes_dict)
    print(wmapes_dict)
    print(crps_dict)

    # save the results, ys, and abnormal to a pickle file
    with open(f'logs//{args.name}_results_{n}.pkl', 'wb') as f:
        pickle.dump({'rmses_dict': rmses_dict, 'maes_dict': maes_dict, 'crps_dict': crps_dict, 'wmapes':wmapes_dict}, f)

    rmses = pd.DataFrame(rmses_dict, index=[0])
    maes = pd.DataFrame(maes_dict, index=[0])
    wmapes = pd.DataFrame(wmapes_dict, index=[0])
    crps = pd.DataFrame(crps_dict, index=[0])

    return rmses, maes, wmapes, crps

#%%
wab_ids = {
    # 'seoul_ABT_concat_MixTruncatedNormal': '7giourqn',
    #        'seoul_ABT_concat_RMSE': '0atfm873',
    #        'seoul_ABT_concat_NB': '743w7y6g',
           # 'seoul_ABT_concat_TruncatedNormal': '3grdaljt',
           # 'seoul_ABT_concat_cross_entropy': 'ag8exyg0',
           # 'guangzhou_ABT_concat_MixTruncatedNormal': '26g9s35f',
           # 'guangzhou_ABT_concat_RMSE': 'a2n8rrtw',
           # 'guangzhou_ABT_concat_NB': 'k5zsvd3d',
           # 'guangzhou_ABT_concat_TruncatedNormal': 'hqlzdz35',
           # 'guangzhou_ABT_concat_cross_entropy': '80us4eox',
           # 'seoul_ABT_new_MixTruncatedNormal': 'ctpd952q',
           'seoul_ABT_new_RMSE': 'w5oefi3j',
           # 'seoul_ABT_new_NB': 'o14igdmj',
           # 'seoul_ABT_new_TruncatedNormal': 'fem5tom1',
           # 'seoul_ABT_new_cross_entropy': 'a4uet0lo',
           # 'guangzhou_ABT_new_MixTruncatedNormal': 'ybwbh0kc',
           # 'guangzhou_ABT_new_MixTruncatedNormal3': 'zun7ap08',
           'guangzhou_ABT_new_RMSE': '1c20716q',
           # 'guangzhou_ABT_new_NB': 'pc0iunwp',
           # 'guangzhou_ABT_new_TruncatedNormal': 'cbm6kvk4',
           # 'guangzhou_ABT_new_cross_entropy': 'znvk9t8t',
           # 'seoul_DeepAR_MixTruncatedNormal': 'doldlrsz',
           # 'seoul_DeepAR_RMSE': 'zvsbtyy2',
           # 'seoul_DeepAR_NB': 't3hpfwkm',
           # 'seoul_DeepAR_TruncatedNormal': 'ioxypofi',
           # 'seoul_DeepAR_cross_entropy': 'jg6e5g2u',
           # 'guangzhou_DeepAR_MixTruncatedNormal': 'mrii22sh',
           # 'guangzhou_DeepAR_RMSE': '856zdwg0',
           # 'guangzhou_DeepAR_NB': 'xoe1a9b4',
           # 'guangzhou_DeepAR_TruncatedNormal': 'hnfceajj',
           # 'guangzhou_DeepAR_cross_entropy': 'cq6b0ocr',
           # 'guangzhou_ABT_cross_entropy_CRPS': '1jbm9clp',

    # 'seoul_ABRotary_concat_RMSE': '08hkm819',
    # 'seoul_ABRotary_concat_NB': 'mii5ne11',
    # 'seoul_ABRotary_concat_TruncatedNormal': '2mfxsjbr',
    # 'seoul_ABRotary_concat_cross_entropy': 'f3osz8fj',
    # 'seoul_ABRotary_concat_MixTruncatedNormal': 'm6mnihqk',

    # 'seoul_ABRotary_half_concat_RMSE': '7ggg6y96',
    # 'seoul_ABRotary_half_concat_NB': '479p2uzs',
    # 'seoul_ABRotary_half_concat_MixTruncatedNormal': 'ab2qxssv',

    # 'guangzhou_ABRotary_concat_RMSE': 'y9f8udxu',
    # 'guangzhou_ABRotary_RMSE': '64flf69z',
    # 'guangzhou_ABRotary_concat_TruncatedNormal': '2f0bpi19',
    # "guangzhou_ABT_inflow": "uhr2fp4h",
    # "guangzhou_ABT_outflow": "1okqimqo",
    # "seoul_ABT_inflow": "it9z5b7z",
    # "seoul_ABT_outflow": "3ujg87b5",
    # "guangzhou_DeepAR_inflow": "cswv4pj2",
    # "guangzhou_DeepAR_outflow": "ln8yas1q",
    # "seoul_DeepAR_inflow": "ktdvrue8",
    # "seoul_DeepAR_outflow": "hixc3060",
    # "seoul_ABT_MixTruncatedNormal": "yodtyil3",
    # "seoul_ABT_CrossEntropy": "xc1an6ql",
    # "seoul_ABT_TruncatedNormal": "izyyuiwd",
    # "Guangzhou_ABT_MixTruncatedNormal": "sabbqn52",
    # "Guangzhou_ABT_CrossEntropy": "r7kv8qwa",
    # "Guangzhou_ABT_TruncatedNormal": "gnsaien5",
           }


results = {}
for model_description, wab_id  in wab_ids.items():
    try:
        rmses, maes, wmapes, crps = evaluate(wab_id, model_description, n=50)
        results[model_description] = {'rmses': rmses, 'maes': maes, 'wmapes': wmapes, 'crps': crps}
    except Exception as e:
        print(f'Error in {model_description}: {e}')
        continue

#%% To see the perstep results
real = y_inflows
forecast = inflows
index = ab_inflows
def get_steps_results(real, forecast, index, crps_result=None):
    results_rmse = []
    results_mae = []
    results_wmape = []
    results_mape = []
    results_crps = []
    rmse = lambda x, y: np.sqrt(np.mean((x-y)**2))
    wmape = lambda x, y: np.sum(np.abs(x-y))/np.sum(y)
    mape = lambda x, y: np.mean(np.abs(x-y)/y)
    mae = lambda x, y: np.mean(np.abs(x-y))

    for step in range(real.shape[1]):
        step_index = index[:, step, :].ravel()
        # step_index = np.arange(real.shape[0])
        step_real = real[step_index, step , :]
        step_forecast = forecast[step_index, step, :]
        rmse_step = rmse(step_real, step_forecast)
        mae_step = mae(step_real, step_forecast)
        wmape_step = wmape(step_real, step_forecast)
        results_rmse.append(rmse_step)
        results_mae.append(mae_step)
        results_wmape.append(wmape_step)
        results_mape.append(mape(step_real, step_forecast))
        if crps_result is not None:
            crps_step = crps_result[step_index, step, :]
            crps_step = crps_step.mean()
            results_crps.append(crps_step)

    return results_rmse, results_mae, results_wmape, results_mape, results_crps

results_rmse, results_mae, results_wmape, results_mape, results_crps = get_steps_results(real, forecast, index, crps_inflows)
# plt.plot(results_wmape, label='wmape')
# plt.plot(results_rmse, label='rmse')
# plt.plot(results_mae, label='mae')
# plt.plot(results_mape, label='mape')
plt.plot(results_crps, label='crps')

#%%
n,m,k = np.where(ab_inflows)
fig, ax = plt.subplots()
cm = ax.scatter(y_inflows[n, m, k], inflows[n, m, k], c=m, cmap='cubehelix',
                edgecolors=None)
# axis equal
ax.set_aspect('equal', 'box')
ax.set_xmargin(0.05)
ax.set_ymargin(0.05)
max_y= y_inflows[n, m, k].max()
ax.plot([0, max_y], [0, max_y], '--')
cbar = fig.colorbar(cm)

#%%
n,m,k = np.where(ab_outflows)
fig, ax = plt.subplots()
cm = ax.scatter(y_outflows[n, m, k], outflows[n, m, k], c=m, cmap='cubehelix_r',
                edgecolors=None)
# axis equal
ax.set_aspect('equal', 'box')
ax.set_xmargin(0.05)
ax.set_ymargin(0.05)
max_y= y_outflows[n, m, k].max()
ax.plot([0, max_y], [0, max_y], '--')
cbar = fig.colorbar(cm)

