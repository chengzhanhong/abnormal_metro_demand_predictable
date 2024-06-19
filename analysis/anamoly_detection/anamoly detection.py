# This script is used to detect abnormal inflow/outflow in the data. The abnormal inflow/outflow is detected by RPCA.
# The outputs are saved to data.csv
import sys
import os
cwd = os.getcwd()
# extend the path to the parent directory to import the modules under the parent directory
sys.path.append(os.path.dirname(os.path.dirname(cwd)))
from utilities.basics import *
import numpy as np
import rpca.ealm
from datasets.datautils import *


def drop_midnight(data, start_minute, end_minute):
    """
    Drop the midnight period of the data.
    data: pd.DataFrame. Indexed by time, wide format data, T x N_stations"""

    data.index.name = 'time'
    data.reset_index(inplace=True)
    minute = (data.time.dt.hour * 60 + data.time.dt.minute).values
    minute[minute == 0] = 1440

    is_running = np.zeros_like(minute)
    is_running[(minute >= start_minute) & (minute < end_minute)] = 1
    data = data[is_running == 1]
    data.set_index('time', inplace=True)

    return data


def get_station_abnormal_index(data, s=14, flow_type="inflow", neighbor=1, iqr_rate=1.5, fig=True):
    """Detect abnormal inflow of station `s` using RPCA
    This function is used to detect abnormal inflow peaks, and return the index of the abnormal peaks.
    neighbor: int, the number of neighbors around an abnormal point that also be marked as abnormal.
    Returns
    -------
    index: np.array, the index of the abnormal peaks.
    """
    # orgainze inflow into a matrix
    other_flow = 'inflow' if flow_type == 'outflow' else 'outflow'
    data_inflow = data.loc[data.station == s, flow_type]
    inflow0 = data_inflow.values
    n = data.time_in_day.nunique()
    n = n*7
    m = inflow0.shape[0]//n
    inflow = inflow0[:m*n].reshape(m, n)  # Here we leave the last few days out

    # RPCA to find abnormal points
    A, E = rpca.ealm.fit(inflow, verbose=False)
    inflow_q25 = np.quantile(inflow, 0.25, axis=0, keepdims=True)
    inflow_q75 = np.quantile(inflow, 0.75, axis=0, keepdims=True)
    inflow_iqr = inflow_q75 - inflow_q25
    idx2 = np.where(np.abs(E) > inflow_iqr*iqr_rate)
    idx = idx2[0]*n + idx2[1] # from 2d index to 1d index
    idx = np.unique(np.concatenate([idx + i for i in range(-neighbor, neighbor + 1)]).ravel())
    idx = idx[(idx >= 0) & (idx < len(inflow0))]
    data_index = data_inflow.index[idx]

    if fig:
        plt.matshow(inflow, aspect='auto')
        plt.plot(idx2[1],idx2[0], 'o',markerfacecolor='none', markeredgecolor='r', markersize=5)

        fig, ax = plt.subplots(figsize=(35, 5))
        outflow0 = data.loc[data.station == s, other_flow].values
        ax.plot(inflow0, label=flow_type)
        ax.plot(outflow0, label='the other flow')
        ax.plot(idx, inflow0[idx], 'ro', label='anomaly')
        ax.legend()
        fig.set_tight_layout(True)
        ax.set_xmargin(0)
    return data_index


def get_all_abnormal_index(data, neighbor=1, flow_type="inflow", **kwargs):
    """Get abnormal index for all stations."""
    index = []
    for s in data['station'].unique():
        index.append(get_station_abnormal_index(data, s=s, neighbor=neighbor, fig=False, flow_type=flow_type))
    index = np.unique(np.concatenate(index).ravel())
    return index


def read_data(args):
    """Return
     data: pd.DataFrame, long format, T*N_stations x 4, columns=['time', 'station', 'inflow', 'outflow']
     """
    start_minute = args.start_minute
    end_minute = args.end_minute
    t_resolution = args.t_resolution

    flow_data_list = []
    for flow in ['inflow', 'outflow']:
        flow_file = args.__dict__[flow + '_file']
        flow_data = pd.read_csv(args.data_path + flow_file, index_col=0, infer_datetime_format=True, parse_dates=[0],
                                dtype=args.default_float)
        flow_data = flow_data.resample(t_resolution).sum()  # resample to t_resolution
        flow_data = drop_midnight(flow_data, start_minute, end_minute)
        flow_data = flow_data.stack()
        flow_data_list.append(flow_data)
    data = pd.concat(flow_data_list, axis=1).reset_index()
    data.columns = ['time', 'station', 'inflow', 'outflow']
    data['station'] = data['station'].astype(args.default_int)
    data.sort_values(['station', 'time'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Add time features
    data['weekday'] = data['time'].dt.weekday.astype(args.default_int)
    minute_in_day = (data['time'].dt.hour * 60 + data['time'].dt.minute).astype(args.default_int)
    # Map time_in_day to unique time index
    unique_minute = minute_in_day.unique()
    unique_minute.sort()
    time_index = np.arange(len(unique_minute))
    time_index = time_index // args.stride
    time_index_map = dict(zip(minute_in_day, time_index))
    data['time_in_day'] = minute_in_day.map(time_index_map).astype(args.default_int)

    # Add if is abnormal flow
    data['abnormal_in'] = False  # abnormal inflow
    data['abnormal_out'] = False  # abnormal outflow
    abnormal_index = get_all_abnormal_index(data, flow_type='inflow', **vars(args))
    data.loc[abnormal_index, 'abnormal_in'] = True
    abnormal_index = get_all_abnormal_index(data, flow_type='outflow', **vars(args))
    data.loc[abnormal_index, 'abnormal_out'] = True
    return data


# %% Test if the anomaly detection works
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args([])
args.torch_int = torch.int32
args.default_int = 'int32'
torch.set_default_dtype(torch.float32)
args.default_float = 'float32'
args.stride = 1
args.dset = "guangzhou"
data_info = data_infos[args.dset]
vars(args).update(data_info)
data = read_data(args)
data.head()
# save to csv
data.to_csv(data_info['data_path'] + 'data.csv',
            index=False)
# data = pd.read_csv(data_info['data_path'] + 'data.csv')

#%% Using manual rules to detect abnormal data
# Guangzhou, s=14, 110, 117, 118, 18
# Seoul, s=18, 27
get_station_abnormal_index(data, s=27, neighbor=0, fig=True, flow_type='inflow', iqr_rate=1.5)
data.time_in_day.unique()