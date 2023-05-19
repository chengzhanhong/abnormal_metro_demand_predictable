import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
import warnings


data_infos = {'guangzhou': {'data_path': '../../../data/GuangzhouMetro//',
                            'inflow_file': 'inflow7_9.csv',
                            'outflow_file': 'outflow7_9.csv',
                            'start_minute': 360,
                            'end_minute': 1440,
                            'subsample_start': '2017-07-01',
                            'subsample_end': '2017-07-20',
                            'prominence': 300,  # used to determine abnormal peaks
                            't_resolution': '10T',  # time resolution used in the model
                            'patch_len': 3,
                            'stride': 3,
                            'num_patch': 36,
                            'target_len': 36,  # 6 hours
                            'flow_diff_r': 0.4, # the maximum possible ratio of in-out flow difference of a day, used to remove outliers
                            },
              'seoul': {'data_path': '../../data/SeoulMetro/',
                        'inflow_file': 'in_data.csv',
                        'outflow_file': 'out_data.csv',
                        'start_minute': 300,
                        'end_minute': 1441,
                        'subsample_start': '2023-01-01',
                        'subsample_end': '2023-01-30',
                        'prominence': 300,  # used to determine abnormal peaks
                        't_resolution': '60T',  # time resolution used in the model
                        'patch_len': 1,
                        'stride': 1,
                        'num_patch': 20,
                        'target_len': 6,  # 6 hours
                        'flow_diff_r': 0.5, # the maximum possible ratio of in-out flow difference of a day, used to remove outliers
                        },
              }


def get_context_len(patch_len, num_patch, stride):
    """Get the context length for forecasting based on patch_len, num_patch, stride."""
    return (num_patch - 1) * stride + patch_len


def unpatch_data(data, stride):
    """unpatch data back into original shape
    data: num_patch x patch_len
    """
    patch_len = data.shape[1]
    effective_len = patch_len - (patch_len - stride)
    part1 = data[:, :effective_len].ravel(order='C')
    part2 = data[-1, -(patch_len - stride):].ravel(order='C')
    return np.concatenate([part1, part2])


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


def get_abnormal_index(data, s=14, prominence=300, patch_len=3, neighbor=1, fig=True):
    """Detect abnormal inflow data using scipy.signal.find_peaks.
    This function is used to detect abnormal inflow peaks, and return the index of the abnormal peaks.
    neighbor: int, the number of neighbors around an abnormal point that also be marked as abnormal.
    """
    # Find peaks
    from scipy.signal import find_peaks
    data0 = data.loc[(data['station'] == s)].copy()
    data0['patch_id'] = np.arange(len(data0)) // patch_len
    data1 = data0.groupby('patch_id').agg({'station': 'first', 'time': 'first', 'inflow': 'sum'}).reset_index()
    data1.sort_values(by='time', inplace=True)
    peaks, _ = find_peaks(data1['inflow'].values, prominence=prominence)

    # Mark outliers
    data1['minute_in_day'] = data1['time'].dt.hour * 60 + data1['time'].dt.minute
    data1['75quantile'] = data1.groupby(['minute_in_day'])[['inflow']].transform('quantile', q=0.75)
    data1['25quantile'] = data1.groupby(['minute_in_day'])[['inflow']].transform('quantile', q=0.25)
    data1['iqr'] = data1['75quantile'] - data1['25quantile']
    data1['upper'] = data1['75quantile'] + 1.5 * data1['iqr']
    abnormal = np.where(data1['inflow'] > data1['upper'])[0]

    # Mark the abnormal peaks
    abnormal_peaks = np.intersect1d(peaks, abnormal)
    # Mark the neighbors of the abnormal peaks
    idx = np.unique(np.concatenate([abnormal_peaks + i for i in range(-neighbor, neighbor + 1)]).ravel())
    idx = idx[(idx >= 0) & (idx < len(data1))]

    index = data.loc[(data['station'] == s) & (data['time'].isin(data1.iloc[idx]['time'])), 'inflow'].index
    # Extend the abnormal index to the whole patch
    index = np.unique(np.concatenate([index + i for i in range(patch_len)]).ravel())
    index = index[(index >= data.index.min()) & (index <= data.index.max())]

    if fig:
        fig, ax = plt.subplots(figsize=(30, 5))
        ax.plot(data0['time'], data0['inflow'])
        ax.plot(data0.loc[index, 'time'], data0.loc[index, 'inflow'], 'ro')
        fig.tight_layout()

    return index


# %%
def detect_anomaly(data, args):
    """Detect anomalies when the difference of daily inflow and outflow larger than args.flow_diff_r,
    or when the total daily flow is zero, mark them as NaN."""
    flow_diff_r = args.flow_diff_r
    daily_inflow = data.groupby([data.time.dt.date, 'station'])[['inflow']].transform('sum').values
    daily_outflow = data.groupby([data.time.dt.date, 'station'])[['outflow']].transform('sum').values
    daily_flow_diff = daily_inflow - daily_outflow
    diff_percent = 2 * np.abs(daily_flow_diff) / (daily_inflow + daily_outflow + 1e-8)
    data.loc[diff_percent.ravel() > flow_diff_r, 'inflow'] = np.nan
    data.loc[(daily_inflow + daily_outflow).ravel() == 0, 'inflow'] = np.nan
    count = np.sum(np.isnan(data.inflow))
    print(f'Marked {count} anomalies ({count / len(data.inflow):.5f}%), will be ignored in training and test.')
    return data


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
        if args.subsample:  # Use subset of data for a quick test
            flow_data = flow_data.loc[args.subsample_start:args.subsample_end, :]
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

    return data


def split_train_val_test(data, args):
    """Split the data into train, val, test set on chronological order.
    The resulting split are rounded to the nearest day.
    train_data are (day start) to (day end)
    val_data and test_data are (day start - context len) to (day end)
    """
    train_r, val_r, test_r = args.train_r, args.val_r, args.test_r
    total_r = train_r + val_r + test_r
    train_r, val_r, test_r = train_r / total_r, val_r / total_r, test_r / total_r

    t_resolution = int(args.t_resolution[:-1])
    day_length = (args.end_minute - args.start_minute) // t_resolution
    data_len = data.time.nunique()
    time_index = np.sort(data.time.unique())

    context_len = args.context_len
    train_size = int(((data_len - context_len) * train_r) // day_length * day_length)
    val_size = int(((data_len - context_len) * val_r) // day_length * day_length)

    train_idx = time_index[:train_size]
    val_idx = time_index[train_size - context_len:train_size + val_size]
    test_idx = time_index[train_size + val_size - context_len:]

    train_data = data.loc[data.time.isin(train_idx), :].reset_index(drop=True)
    val_data = data.loc[data.time.isin(val_idx), :].reset_index(drop=True)
    test_data = data.loc[data.time.isin(test_idx), :].reset_index(drop=True)

    return train_data, val_data, test_data


class MetroDataset(Dataset):
    def __init__(self, data, mask_method, args):
        self.data = data.sort_values(['station', 'time']).reset_index(drop=True)
        self.patch_len = args.patch_len
        self.num_patch = args.num_patch
        self.stride = args.stride
        self.context_len = get_context_len(args.patch_len, args.num_patch, args.stride)
        self.target_len = args.target_len
        self.num_target_patch = self.target_len // self.patch_len
        self.sample_len = self.context_len + self.target_len

        self.f_index, self.if_index = self.get_feasible_index()  # f_index is the union of normal and abnormal index
        self.normal_index = self.f_index
        self.abnormal_index = np.array([])
        self._index = self.f_index  # Default f_index is the self._index
        self.abnormal_station = None

        self.flow_type = torch.concat((torch.zeros(self.num_patch, dtype=args.torch_int),
                                       torch.ones(self.num_patch, dtype=args.torch_int),
                                       torch.ones(self.num_target_patch, dtype=args.torch_int) * 2))
        self.num_flow_type = 3
        self.num_station = self.data.station.nunique()
        self.num_weekday = 7
        self.mask_method = mask_method
        self.mask_ratio = args.data_mask_ratio

        # number of time_in_day
        time_in_day = self.data.loc[self.f_index, 'time'].dt.minute + \
                      self.data.loc[self.f_index, 'time'].dt.hour * 60
        self.num_time_in_day = len(time_in_day.unique())
        t_resolution = int(args.t_resolution[:-1])
        if self.num_time_in_day > (time_in_day.max() - time_in_day.min()) / t_resolution / self.stride + 1:
            warnings.warn(f'num_time_in_day {self.num_time_in_day}, please check the stride, patch_len are correct.')

    def __len__(self):
        return len(self._index)

    def exclude_abnormal_flow(self, station=None, prominence=300):
        if station is not None:
            abnormal_index = get_abnormal_index(self.data, station, prominence=prominence, patch_len=self.patch_len)
            idx_set = set()
            for idx in abnormal_index:
                idx_set.update(range(idx - self.sample_len + 1, idx + 1))
            self.abnormal_index = np.array(list(idx_set))
            self.abnormal_index = np.setdiff1d(self.abnormal_index, self.if_index)
            self.normal_index = np.setdiff1d(self.f_index, self.abnormal_index)
            self.abnormal_station = station
        self._index = self.normal_index

    def only_abnormal_flow(self, station=None, prominence=300):
        if station is not None:
            abnormal_index = get_abnormal_index(self.data, station, prominence=prominence, patch_len=self.patch_len)
            idx_set = set()
            for idx in abnormal_index:
                idx_set.update(range(idx - self.sample_len + 1, idx + 1))
            self.abnormal_index = np.array(list(idx_set))
            self.abnormal_index = np.setdiff1d(self.abnormal_index, self.if_index)
            self.normal_index = np.setdiff1d(self.f_index, self.abnormal_index)
            self.abnormal_station = station
        self._index = self.abnormal_index

    def get_feasible_index(self):
        """Get feasible index with nan and station switch points excluded.
        And also at the start of integer path_size
        """
        feasible_idx = self.data.index.values
        feasible_idx = feasible_idx[:-self.sample_len + 1]
        feasible_idx = feasible_idx[::self.stride]

        nan_idx = np.where(np.isnan(self.data.inflow.values))[0]
        new_station_idx = np.where(self.data.station.values != self.data.station.shift(1).values)[0]

        infeasible_idx = set()
        for idx in nan_idx:
            infeasible_idx.update(range(idx - self.sample_len + 1, idx + 1))
        for idx in new_station_idx:
            infeasible_idx.update(range(max(idx - self.sample_len, 0), idx))
        infeasible_idx = np.array(list(infeasible_idx))

        feasible_idx = np.setdiff1d(feasible_idx, infeasible_idx)
        infeasible_idx = np.setdiff1d(self.data.index.values, feasible_idx)
        return feasible_idx, infeasible_idx

    def __getitem__(self, index):
        """Returns
         context: num_patch*2 x patch_len
         flow_type: num_patch*2
         station: num_patch*2
         weekday: num_patch*2
         time_in_day: num_patch*2
         target: target_len
         """
        data_piece = self.data.iloc[self._index[index]:self._index[index] + self.sample_len, :]
        outflow_data = torch.from_numpy(data_piece.outflow.values[:self.context_len]).unfold(0, self.patch_len,
                                                                                             self.stride)
        inflow_data = torch.from_numpy(data_piece.inflow.values).unfold(0, self.patch_len, self.stride)
        inflow_data, target, fcst_loc = self.random_mask(inflow_data, method=self.mask_method,
                                                         mask_ratio=self.mask_ratio)
        data = torch.cat((outflow_data, inflow_data), dim=0)
        fcst_loc += self.num_patch

        features = torch.from_numpy(data_piece[['station', 'weekday', 'time_in_day']].values[:: self.stride])
        station = torch.cat((features[0:self.num_patch, 0], features[:, 0]), dim=0)
        weekday = torch.cat((features[0:self.num_patch, 1], features[:, 1]), dim=0)
        time_in_day = torch.cat((features[0:self.num_patch, 2], features[:, 2]), dim=0)

        flow_type = self.flow_type.clone()
        flow_type[fcst_loc] = self.num_flow_type - 1

        return (data, flow_type, station, weekday, time_in_day, fcst_loc), target

    def random_mask(self, data, method='target', mask_ratio=0.2):
        """Randomly mask the data with mask_ratio.
        method: 'target' or 'both'
        mask_ratio: float, the ratio of masked data in the context, does not work for 'target' method.
        data: the data to be masked (inflow only), shape: (num_patch+num_target_patch, patch_len)
        return
        -------
        masked_data: masked data.
        target: target data.
        fcst_loc: the location of the target data.
        """
        masked_data = data.clone()
        n = data.shape[0]

        if method == "target":
            fcst_loc = torch.arange(self.num_patch, n)
        elif method == 'both':
            fcst_loc = torch.concat([torch.randperm(self.num_patch)[:int((self.num_patch) * mask_ratio)],
                                     torch.arange(self.num_patch, n)])
        else:
            raise ValueError('method must be "target" or "both".')

        target = data[fcst_loc, :]
        masked_data[fcst_loc, :] = 0
        return masked_data, target, fcst_loc

    def get_data_from_ts(self, time, station, method='target', mask_ratio=0.2):
        """Get data from time and station."""
        # Test whether the time and station is feasible and valid.
        index_now = self.data[(self.data.time == time) & (self.data.station == station)].index.values[0]
        index_start = index_now - self.context_len
        if index_start not in self._index:
            print('Invalid or infeasible time and station.')
            return None

        data_piece = self.data.iloc[index_start:index_now + self.target_len, :]

        # Get the location of index in self._index.
        index = np.where(self._index == index_start)[0][0]

        old_method = self.mask_method
        old_mask_ratio = self.mask_ratio

        self.mask_method = method
        self.mask_ratio = mask_ratio

        x, y = self.__getitem__(index)

        self.mask_method = old_method
        self.mask_ratio = old_mask_ratio
        return (x, y), data_piece
