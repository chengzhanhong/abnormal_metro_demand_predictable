import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler

data_infos = {'guangzhou': {'inflow_file': 'inflow7_9.csv',
                            'outflow_file': 'outflow7_9.csv',
                            'start_minute': 360,
                            'end_minute': 1440,
                            'subsample_start': '2017-07-01',
                            'subsample_end': '2017-07-07'}
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


def group_midnight(data, start_minute, end_minute):
    """
    Return a dataframe where the midnight rows of a day is sum into a single row.
    data: pd.DataFrame. Indexed by time, wide format data, T x N_stations"""
    stations = data.columns
    data.index.name = 'time'
    data.reset_index(inplace=True)
    minute = (data.time.dt.hour * 60 + data.time.dt.minute).values
    minute[minute == 0] = 1440  # Set 00:00 to the end of the day

    is_running = np.zeros_like(minute)
    is_running[(minute >= start_minute) & (minute <= end_minute)] = 1
    time_index = np.cumsum(is_running)  # Thus midnight period of the same day has the same time_index

    functions = {station: 'sum' for station in stations}
    functions['time'] = 'first'
    data = data.groupby(time_index).agg(functions)
    data.set_index('time', inplace=True)
    return data


def detect_anomaly(data, args):
    """Detect anomalies when the difference of daily inflow and outflow larger than args.flow_diff_r,
    or when the total daily flow is zero, mark them as NaN."""
    flow_diff_r = args.flow_diff_r
    daily_inflow = data.groupby([data.time.dt.date, 'station'])[['inflow']].transform('sum').values
    daily_outflow = data.groupby([data.time.dt.date, 'station'])[['outflow']].transform('sum').values
    daily_flow_diff = daily_inflow - daily_outflow
    diff_percent = 2 * np.abs(daily_flow_diff) / (daily_inflow + daily_outflow + 1e-8)
    data.loc[diff_percent.ravel() > flow_diff_r, 'inflow'] = np.nan
    data.loc[(daily_inflow+daily_outflow).ravel() == 0, 'inflow'] = np.nan
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
        if args.subsample:  # For quick test
            flow_data = flow_data.loc[args.subsample_start:args.subsample_end, :]
        flow_data = group_midnight(flow_data, start_minute, end_minute)
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
    time_index_map = dict(zip(minute_in_day, time_index))
    data['time_in_day'] = minute_in_day.map(time_index_map).astype(args.default_int)

    return data


def split_train_val_test(data, args):
    """Split the data into train, val, test set on chronological order. The split could be in the middle of a day."""
    train_r, val_r, test_r = args.train_r, args.val_r, args.test_r
    total_r = train_r + val_r + test_r
    train_r, val_r, test_r = train_r / total_r, val_r / total_r, test_r / total_r

    data_len = data.time.nunique()
    context_len = args.context_len
    train_size = int((data_len - context_len) * train_r)
    val_size = int((data_len - context_len) * val_r)

    time_index = np.sort(data.time.unique())
    train_idx = time_index[:train_size + context_len]
    val_idx = time_index[train_size:train_size + val_size + context_len]
    test_idx = time_index[train_size + val_size:]

    train_data = data.loc[data.time.isin(train_idx), :].reset_index(drop=True)
    val_data = data.loc[data.time.isin(val_idx), :].reset_index(drop=True)
    test_data = data.loc[data.time.isin(test_idx), :].reset_index(drop=True)

    return train_data, val_data, test_data


def get_data_loader(args):
    # add data info to args
    data_info = data_infos[args.dset]
    for key, value in data_info.items():
        setattr(args, key, value)

    dataset = MetroDataset(args)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)
    return data_loader


class MetroDataset(Dataset):
    def __init__(self, data, args):
        self.data = data.sort_values(['station', 'time']).reset_index(drop=True)
        self.patch_len = args.patch_len
        self.num_patch = args.num_patch
        self.stride = args.stride
        self.context_len = get_context_len(args.patch_len, args.num_patch, args.stride)
        self.target_len = args.target_len
        self.sample_len = self.context_len + self.target_len
        self.f_index, self.if_index = self.get_feasible_index()
        self.flow_type = torch.concat((torch.zeros(self.num_patch, dtype=args.torch_int),
                                       torch.ones(self.num_patch, dtype=args.torch_int)))
        self.num_flow_type = 2
        self.num_station = self.data.station.nunique()
        self.num_weekday = 7
        self.num_time_in_day = self.data.time_in_day.nunique()

    def __len__(self):
        return len(self.f_index)

    def get_feasible_index(self):
        """Get feasible index with nan and station switch points excluded."""
        feasible_idx = self.data.index.values
        feasible_idx = feasible_idx[:-self.sample_len + 1]

        nan_idx = np.where(np.isnan(self.data.inflow.values))[0]
        new_station_idx = np.where(self.data.station.values != self.data.station.shift(1).values)[0]


        infeasible_idx = set()
        for idx in nan_idx:
            infeasible_idx.update(range(idx - self.sample_len + 1, idx + 1))
        for idx in new_station_idx:
            infeasible_idx.update(range(max(idx - self.sample_len, 0), idx))
        infeasible_idx = np.array(list(infeasible_idx))

        feasible_idx = np.setdiff1d(feasible_idx, infeasible_idx)
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
        data_piece = self.data.iloc[self.f_index[index]:self.f_index[index] + self.sample_len, :]
        context = torch.from_numpy(data_piece.outflow.values[:self.context_len]).unfold(0, self.patch_len, self.stride)
        context = torch.cat((context,
                             torch.from_numpy(data_piece.inflow.values[:self.context_len]).unfold(0, self.patch_len,
                                                                                                  self.stride)), dim=0)

        station = torch.from_numpy(np.repeat(data_piece.station.values[0], self.num_patch * 2))
        weekday = torch.from_numpy(np.tile(data_piece.weekday.values[:self.context_len: self.stride], 2))
        time_in_day = torch.from_numpy(np.tile(data_piece.time_in_day.values[:self.context_len: self.stride], 2))

        target = torch.from_numpy(data_piece.inflow.values[self.context_len:])

        return (context, self.flow_type, station, weekday, time_in_day), target

    def get_data_from_ts(self, time, station):
        """Get data from time and station."""
        # Test whether the time and station is feasible and valid.
        index_now = self.data[(self.data.time == time) & (self.data.station == station)].index.values[0]
        index_start = index_now - self.context_len
        if index_start not in self.f_index:
            print('Invalid or infeasible time and station.')
            return None

        data_piece = self.data.iloc[index_start:index_now + self.target_len, :]

        # Get the location of index in f_index.
        index_ = np.where(self.f_index == index_start)[0][0]

        return self.__getitem__(index_), data_piece
