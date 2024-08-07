# %%
import sys
import os

cwd = os.getcwd()
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.basics import *
from datasets.datautils import *
from models.ABT_concat import ABTransformer_concat
from models.ABT_new import ABTransformer
from models.DeepAR import DeepAR
from models.basics import *
import seaborn as sns
from sklearn.cluster import KMeans
sns.set_palette("deep")
plt.rcParams.update({'ytick.direction': 'in', 'xtick.direction': 'in'})

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(device)
else:
    print('No GPU available, using the CPU instead.')

#%% Load the data and the model for analysis
def load_data(wab_id):
    args = get_wandb_args(f'deepzhanhong/ABTransformer/{wab_id}')
    args.initial_num_bins = 1024
    args.data_path = args.data_path[3:]
    args.attn_mask_type = 'time'
    if 'forecast_target' not in args:
        args.forecast_target = 'both'

    if 'sample_divide' in args:
        if args.sample_divide > 1:
            args.num_patch = args.num_patch//2
            args.num_target_patch = args.num_target_patch//2
            args.input_len = args.input_len//2
            args.target_len = args.target_len//2

    data = pd.read_csv(args.data_path[3:]+'data.csv', parse_dates=['time'], dtype={'station': args.default_int,
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
    test_dataset.test_mode = True

    x_loc, x_scale = get_loc_scale(train_dataset.data.loc[train_dataset.f_index, :], args.standardization)
    x_loc = torch.tensor(x_loc, dtype=torch.float32).to(device)
    x_scale = torch.tensor(x_scale, dtype=torch.float32).to(device)
    attn_mask = get_attn_mask(args.num_patch, args.num_target_patch-1, device=device,
                              type=args.attn_mask_type)
    return dataset, x_loc, x_scale, attn_mask, args

def load_model(model_id, x_loc, x_scale, attn_mask, device):
    models = {'ABT': ABTransformer, 'ABT_concat': ABTransformer_concat, 'DeepAR': DeepAR, 'ABT_new': ABTransformer, 'ABT2': ABTransformer}
    paths = {'ABT': 'exps\\ABTransformer\\log\\', 'ABT_concat': 'exps\\ABTransformer\\log\\',
             'DeepAR': 'exps\\DeepAR\\log\\', 'ABT_new': 'exps\\ABTransformer\\log\\',
             'ABT2': 'exps\\ABTransformer\\log\\'}

    args = get_wandb_args(f'deepzhanhong/ABTransformer/{model_id}')
    args.initial_num_bins = 1024
    args.data_path = args.data_path[3:]
    args.attn_mask_type = 'time'
    if 'forecast_target' not in args:
        args.forecast_target = 'both'
    args.bin_edges = np.array(eval(args.bin_edges.replace('\n', '').replace(' ', ',')))

    if 'sample_divide' in args:
        if args.sample_divide > 1:
            args.num_patch = args.num_patch//2
            args.num_target_patch = args.num_target_patch//2
            args.input_len = args.input_len//2
            args.target_len = args.target_len//2

    model = models[args.model](x_loc=x_loc, x_scale=x_scale, attn_mask=attn_mask, **vars(args))
    model.to(device)
    if 'pe' in args:
        if args.pe == 'rotary':
            model.backbone.W_pos = None
    model.load_state_dict(torch.load(f'{paths[args.model]}{args.name}.pth'), strict=False)
    return model

dataset2, x_loc2, x_scale2, attn_mask2, args2 = load_data('uogg4iwe')  # seoul

#%% Plot the forecast clustering results
# id  = '99qguiit'
# # t = '2017-09-08 20:30:00'
# # s = 13
# t = '2017-09-23 21:00:00'
# s = 33  # 110， 33
#
# dataset1, x_loc1, x_scale1, attn_mask1, args1 = load_data(id)  # guangzhou
# model1 = load_model(id, x_loc1, x_scale1, attn_mask1, device) # guangzhou, '7vwauu8u' is the used deterministic
# t1 = pd.Timestamp(t)
# (x, y, _), data_piece = dataset1.get_data_from_ts(t1, s)
# model = model1
#
# ll = data_piece.shape[0]
# x = [xx.unsqueeze(0) for xx in x]
# x = [xx.to(device) for xx in x]
# model.eval()
# y_hat = model.forecast_samples(x, n=500)
# y_hat = y_hat.squeeze().detach().cpu().numpy()
# inflow = y_hat[:, :, 0]
# outflow = y_hat[:, :, 1]
# ll_f = inflow.shape[1]
# #%%
#
# labels = ['outflow', 'inflow']
# colors = ['C1', 'C0']
# forecasts = [outflow, inflow]
# fig = plt.figure(figsize=(6, 6))
# axes = [[],[]]
# grid =fig.add_gridspec(2, 1, hspace=0.2, left=0.1, right=0.95, top=0.95, bottom=0.07)
# for i in range(2):
#     inner_grid = grid[i].subgridspec(2, 1, hspace=0)
#     for j in range(2):
#         axes[i].append(fig.add_subplot(inner_grid[j]))
#         axes[i][j].plot(data_piece[labels[j]].values[-2*ll_f:], color=colors[j])
#         if i == 0:
#             axes[0][j].plot(range(ll_f,2*ll_f), forecasts[j].T, color='gray', alpha=0.02, zorder=-100)
#             q05 = np.quantile(forecasts[j], 0.05, axis=0)
#             q95 = np.quantile(forecasts[j], 0.95, axis=0)
#             q25 = np.quantile(forecasts[j], 0.25, axis=0)
#             q75 = np.quantile(forecasts[j], 0.75, axis=0)
#             axes[i][j].fill_between(range(ll_f,2*ll_f), q05, q95, color=colors[j], alpha=0.3, label='90% quantile')
#             axes[i][j].fill_between(range(ll_f,2*ll_f), q25, q75, color=colors[j], alpha=0.5, label='50% quantile')
#
# # hdb_in = HDBSCAN(min_cluster_size=5, metric='correlation')
# # hdb_out = HDBSCAN(min_cluster_size=5, metric='correlation', allow_single_cluster=True)
#
# cmodel = KMeans(n_clusters=4, random_state=11)
#
# data = np.concatenate([inflow, outflow], axis=1)
# cmodel.fit(data/(np.sqrt((data**2).sum(axis=1, keepdims=True))))
# for i in range(2):
#     axes[1][i].plot(range(ll_f,2*ll_f), forecasts[i].T, color='gray', alpha=0.05, zorder=-100)
#     for j, label in enumerate(np.unique(cmodel.labels_)):
#         if label == -1:
#             continue
#         idx = (cmodel.labels_==label)
#         print(f'Cluster {label}: {sum(idx)} samples')
#         q05 = np.quantile(forecasts[i][idx], 0.05, axis=0)
#         q95 = np.quantile(forecasts[i][idx], 0.95, axis=0)
#         q25 = np.quantile(forecasts[i][idx], 0.25, axis=0)
#         q75 = np.quantile(forecasts[i][idx], 0.75, axis=0)
#         axes[1][i].fill_between(range(ll_f,2*ll_f), q05, q95, color=f'C{j}', alpha=0.1)
#         axes[1][i].fill_between(range(ll_f,2*ll_f), q25, q75, color=f'C{j}', alpha=0.3)

#%%
def plot_probablistic_forecast(id, t, s, axes, n_sample=500, N_clusters=1, plot_samples=False, x_interval=4,
                               title='', pp=1, sample_pp=0.2, alpha=0.1):
    dataset, x_loc, x_scale, attn_mask, args1 = load_data(id)  # guangzhou
    model = load_model(id, x_loc, x_scale, attn_mask, device) # guangzhou, '7vwauu8u' is the used deterministic
    t1 = pd.Timestamp(t)
    (x, y, ab_idx), data_piece = dataset.get_data_from_ts(t1, s)

    x = [xx.unsqueeze(0) for xx in x]
    x = [xx.to(device) for xx in x]
    model.eval()
    y_hat = model.forecast_samples(x, n=n_sample)
    y_hat = y_hat.squeeze().detach().cpu().numpy()
    inflow = y_hat[:, :, 0]
    outflow = y_hat[:, :, 1]
    ll_f = inflow.shape[1]

    colors = ['C1', 'C0']
    labels = ['outflow', 'inflow']
    forecasts = [outflow, inflow]
    if N_clusters == 1:  # If no cluster
        for i in range(2):
            axes[i].plot(data_piece[labels[i]].values[-(pp+1)*ll_f:], color=colors[i])  # Real data
            if plot_samples:
                nn = int(n_sample*sample_pp)
                idx = np.random.choice(n_sample, nn, replace=False)
                axes[i].plot(range(pp*ll_f,(pp+1)*ll_f), forecasts[i][idx].T, color='gray', alpha=alpha, zorder=-100)
            q05 = np.quantile(forecasts[i], 0.05, axis=0)
            q95 = np.quantile(forecasts[i], 0.95, axis=0)
            q25 = np.quantile(forecasts[i], 0.25, axis=0)
            q75 = np.quantile(forecasts[i], 0.75, axis=0)
            axes[i].fill_between(range(pp*ll_f,(pp+1)*ll_f), q05, q95, color=colors[i], alpha=0.3, label='90% quantile')
            axes[i].fill_between(range(pp*ll_f,(pp+1)*ll_f), q25, q75, color=colors[i], alpha=0.5, label='50% quantile')

    elif N_clusters > 1:
        cmodel = KMeans(n_clusters=N_clusters, random_state=11)
        data = np.concatenate([inflow, outflow], axis=1)
        cmodel.fit(data/(np.sqrt((data**2).sum(axis=1, keepdims=True))))
        for i in range(2):
            axes[i].plot(data_piece[labels[i]].values[-(pp+1)*ll_f:], color=colors[i])  # Real data
            if plot_samples:
                axes[i].plot(range(pp*ll_f,(pp+1)*ll_f), forecasts[i].T, color='gray', alpha=alpha, zorder=-100)
            for j, label in enumerate(np.unique(cmodel.labels_)):
                idx = (cmodel.labels_==label)
                print(f'Cluster {label}: {sum(idx)} samples')
                q05 = np.quantile(forecasts[i][idx], 0.05, axis=0)
                q95 = np.quantile(forecasts[i][idx], 0.95, axis=0)
                q25 = np.quantile(forecasts[i][idx], 0.25, axis=0)
                q75 = np.quantile(forecasts[i][idx], 0.75, axis=0)
                axes[i].fill_between(range(pp*ll_f,(pp+1)*ll_f), q05, q95, color=f'C{j+2}', alpha=0.1)
                axes[i].fill_between(range(pp*ll_f,(pp+1)*ll_f), q25, q75, color=f'C{j+2}', alpha=0.3)
        p = np.bincount(cmodel.labels_)/len(cmodel.labels_)
        for i in range(N_clusters):
            axes[1].text(0.02, 0.9-i*0.1, f'Cluster {i}: p={p[i]:.2f}',
                         fontsize=12, transform=axes[1].transAxes, color=f'C{i+2}')
    axes[1].plot(np.arange((pp+1)*ll_f)[ab_idx[-(pp+1)*ll_f:,0]],
                 data_piece['inflow'].values[-(pp+1)*ll_f:][ab_idx[-(pp+1)*ll_f:,0]], 'o', color='C3',
                 label='Abnormal points', markerfacecolor='none')
    axes[0].plot(np.arange((pp+1)*ll_f)[ab_idx[-(pp+1)*ll_f:,1]],
                 data_piece['outflow'].values[-(pp+1)*ll_f:][ab_idx[-(pp+1)*ll_f:,1]], 'o', color='C3',
                 markerfacecolor='none')

    axes[0].sharex(axes[1])
    axes[0].set_xticklabels([])
    axes[0].label_outer()
    axes[0].text(0.02, 0.9, title, fontsize=12, transform=axes[0].transAxes)
    axes[1].set_xticks(range(0, (pp+1)*ll_f, x_interval))
    xlabels = data_piece['time'].dt.strftime('%H:%M').values[-(pp+1)*ll_f::x_interval]
    axes[1].set_xticklabels(xlabels)
    axes[0].set_xmargin(0)
    axes[1].set_xmargin(0)
    # set y axis starts from 0
    axes[0].set_ylim(-10, axes[0].get_ylim()[1]*1.08)
    axes[1].set_ylim(-10, axes[1].get_ylim()[1]*1.08)
    # add grid
    axes[0].grid(which='major', linestyle='--', linewidth='0.5', color='gray')
    axes[1].grid(which='major', linestyle='--', linewidth='0.5', color='gray')

#%%
fig = plt.figure(figsize=(10, 10))
axes = []
grid =fig.add_gridspec(2, 2, hspace=0.1, left=0.1, right=0.95, top=0.95, bottom=0.07)
for i in range(2):
    for j in range(2):
        inner_grid = grid[i,j].subgridspec(2, 1, hspace=0)
        ax0 = fig.add_subplot(inner_grid[0])
        ax1 = fig.add_subplot(inner_grid[1])
        axes.append([ax0, ax1])
# 99qguiit
# ya0ce42r

# gnsaien5
# izyyuiwd
plot_probablistic_forecast('klzimz7g', t = '2017-09-20 17:00:00', s=44, axes=axes[0], N_clusters=1, plot_samples=True,
                           title='(A) Guangzhou Zhujiang New Town', x_interval=12, pp=2,alpha=0.1,sample_pp=0.2)
plot_probablistic_forecast('klzimz7g', t = '2017-09-23 21:00:00', s=33, axes=axes[1], N_clusters=1, plot_samples=True,
                           title='(B) Guangzhou Baiyun Park', x_interval=12, pp=2,alpha=0.1,sample_pp=0.2)
plot_probablistic_forecast('ya0ce42r', t = '2023-06-22 18:00:00', s=27, axes=axes[2], N_clusters=1, plot_samples=True,
                           title='(C-1) Seoul Sport Complex', x_interval=3, pp=2, alpha=0.1, sample_pp=0.2)
plot_probablistic_forecast('ya0ce42r', t = '2023-06-22 18:00:00', s=27, axes=axes[3], N_clusters=3,
                           title='(C-2) Seoul Sport Complex', x_interval=3, pp=2)
axes[0][0].set_ylabel('Alighting P / 15 min')
axes[2][0].set_ylabel('Alighting P / hour')
axes[0][1].set_ylabel('Boarding P / 15 min')
axes[2][1].set_ylabel('Boarding P / hour')

# axes[1][1].set_ylim(-10, 4200)
# axes[2][1].set_ylim(-10, 8200)
# axes[3][1].set_ylim(-10, 8200)


fig.savefig('figs/forecast_cluster_1.pdf', bbox_inches='tight')
#%%
y = 60
a = 0
for i in range(0, y, 3):
    a += 11500 + i*1500

print(a)