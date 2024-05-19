# The new data utilities for metro datasets.
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
import warnings
import random

data_infos = {'guangzhou': {'data_path': '../../data/GuangzhouMetro//',
                            'inflow_file': 'in_data.csv',
                            'outflow_file': 'out_data.csv',
                            'start_minute': 360,
                            'end_minute': 1440,
                            't_resolution': '10T',  # time resolution used in the model
                            'input_len': 108,
                            'target_len': 36,  # 6 hours
                            'test_date_start': '2017-09-15',  # equivalent to 2017-09-16
                            'test_date_end': '2017-10-01',
                            'val_date_start': '2017-09-08',  # equivalent to 2017-09-09
                            'val_date_end': '2017-09-15',
                            'sample_interval':2,  # The skip interval for fast testing
                            'patch_len':3
                            },
              'seoul': {'data_path': '../../data/SeoulMetro/',
                        'inflow_file': 'in_data.csv',
                        'outflow_file': 'out_data.csv',
                        'start_minute': 240,
                        'end_minute': 1381,
                        't_resolution': '60T',  # time resolution used in the model
                        'input_len': 20,
                        'target_len': 6,  # 6 hours
                        'test_date_start': '2023-06-09',  # equivalent to 2023-06-10
                        'test_date_end': '2023-07-01',
                        'val_date_start': '2023-05-31',  # equivalent to 2023-06-01
                        'val_date_end': '2023-06-09',
                        'sample_interval': 1,  # The skip interval for fast testing
                        'patch_len':1
                        },
              }

model_infos = {'DeepAR': {'d_model': 128,
                          'n_layers': 3,
                          'dropout': 0.05,
                          'n_epochs': 20,
                          'patience': 5,
                          'max_lr': 0.001,
                          'anneal_strategy': 'linear',
                          'datatype': 'DeepAR',
                          },
               'ABT_concat': {'d_model': 128,
                              'n_heads': 8,
                              'n_layers': 3,
                              'dropout': 0.05,
                              'attn_dropout': 0,
                              'n_epochs': 20,
                              'patience': 5,
                              'max_lr': 0.001,
                              'anneal_strategy': 'linear',
                              'datatype': 'ABT_concat',
                              'station_eb': True,
                              'weekday_eb': True,
                              'time_eb': True,
                              'seed': 0,
                              'max_run_time': 10,
                                },
               'ABT': {'d_model': 128,
                       'n_heads': 8,
                       'n_layers': 3,
                       'dropout': 0.05,
                       'attn_dropout': 0,
                       'n_epochs': 20,
                       'patience': 5,
                       'max_lr': 0.001,
                       'anneal_strategy': 'linear',
                       'datatype': 'ABT',
                       'station_eb': True,
                       'weekday_eb': True,
                       'time_eb': True,
                       'seed': 0,
                       'max_run_time': 10,
                       }
                }

def reset_random_seeds(n=1):
    os.environ['PYTHONHASHSEED'] = str(n)
    # tf.random.set_seed(n)
    torch.random.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def unpatch_data(data, stride):
    """unpatch data back into original shape
    data: num_patch x patch_len
    """
    patch_len = data.shape[1]
    effective_len = patch_len - (patch_len - stride)
    part1 = data[:, :effective_len].ravel(order='C')
    part2 = data[-1, -(patch_len - stride):].ravel(order='C')
    return np.concatenate([part1, part2])


def get_train_val_test_index(data, args):
    """Split the data into train, val, test set according to the time, then return the index.
    """
    time_index = np.sort(data.time.unique())

    # Get test idx
    test_date_start = pd.to_datetime(args.test_date_start)
    test_date_end = pd.to_datetime(args.test_date_end)
    test_idx = time_index[(time_index >= test_date_start) & (time_index <= test_date_end)]
    test_idx = data.loc[data.time.isin(test_idx)].index

    # Get val idx
    val_date_start = pd.to_datetime(args.val_date_start)
    val_date_end = pd.to_datetime(args.val_date_end)
    val_idx = time_index[(time_index >= val_date_start) & (time_index <= val_date_end)]
    val_idx = data.loc[data.time.isin(val_idx)].index

    # Get train idx by removing val_idx and test_idx
    train_idx = np.setdiff1d(data.index, np.concatenate([val_idx, test_idx]))

    return train_idx, val_idx, test_idx


class MetroDataset_base(Dataset):
    def __init__(self, data, args, f_index=None):
        self.data = data
        self.patch_len = args.patch_len
        self.sample_interval = args.sample_interval
        self.input_len = args.input_len  # length of input sequence = num input patch * patch_len
        self.target_len = args.target_len # length of target sequence = num target patch * patch_len
        self.sample_len = self.input_len + self.target_len

        self.num_input_patch = (args.input_len - args.patch_len) // args.patch_len + 1
        self.num_sample_patch = (self.sample_len - args.patch_len) // args.patch_len + 1

        self.station_eb = args.station_eb
        self.weekday_eb = args.weekday_eb
        self.time_eb = args.time_eb

        if f_index is None:
            self.f_index, _ = self.get_feasible_index()
        else:
            self.f_index = f_index
        self._index = self.f_index

        self.num_station = self.data.station.nunique()
        self.num_weekday = 7
        # number of time_in_day
        self.num_time_in_day = self.data['time_in_day'].nunique()

    def __len__(self):
        return len(self._index)

    def get_feasible_index(self):
        """Get feasible index with nan and station switch points excluded.
        And also at the start of integer path_size
        """
        feasible_idx = self.data.index.values
        feasible_idx = feasible_idx[:-self.sample_len + 1:self.sample_interval]

        nan_idx = np.where(np.isnan(self.data.inflow.values))[0]
        if len(nan_idx) > 0:
            raise ValueError('There are nan values in the data.')

        new_station_idx = np.where(self.data.station.values != self.data.station.shift(1).values)[0]
        infeasible_idx = set()
        for idx in new_station_idx:
            infeasible_idx.update(range(max(idx - self.sample_len, 0), idx))
        infeasible_idx = np.array(list(infeasible_idx))

        feasible_idx = np.setdiff1d(feasible_idx, infeasible_idx)
        infeasible_idx = np.setdiff1d(self.data.index.values, feasible_idx)
        return feasible_idx, infeasible_idx


    def get_data_from_ts(self, time, station):
        """Get data from time and station."""
        # Test whether the time and station is feasible and valid.
        index_now = self.data[(self.data.time == time) & (self.data.station == station)].index.values[0]
        index_start = index_now - self.input_len
        if index_start not in self._index:
            print('Invalid or infeasible time and station.')
            return None

        data_piece = self.data.iloc[index_start:index_now + self.target_len, :]

        # Get the location of index in self._index.
        index = np.where(self._index == index_start)[0][0]
        x, y, is_abnormal = self.__getitem__(index)
        return (x, y, is_abnormal), data_piece


class MetroDataset_total(MetroDataset_base):
    def __init__(self, data, args, f_index=None, datatype='seq'):
        self.data = data.sort_values(['station', 'time']).reset_index(drop=True)
        super(MetroDataset_total, self).__init__(self.data, args, f_index)

        datatype_dict = {'ABT': MetroDataset_deepar,
                         'DeepAR': MetroDataset_deepar,
                         'ABT_concat': MetroDataset_deepar  # ABT_concat uses the same data structure as DeepAR
                         }
        child_dataset = datatype_dict[datatype]

        # Get train, val, test Dataset
        self.train_idx, self.val_idx, self.test_idx = get_train_val_test_index(data, args)
        self.TrainDataset = child_dataset(data, args, np.intersect1d(self.train_idx, self.f_index))
        self.ValDataset = child_dataset(data, args, np.intersect1d(self.val_idx, self.f_index))
        self.TestDataset = child_dataset(data, args, np.intersect1d(self.test_idx, self.f_index))

    def get_data_from_ts(self, time, station, method='target', mask_ratio=0.2):
        """Get input and target from a specific time and station."""
        # Test whether the time and station is feasible and valid.
        index_now = self.data[(self.data.time == time) & (self.data.station == station)].index.values[0]
        index_start = index_now - self.input_len

        # # Get the location of index in self._index.
        # index = np.where(self._index == index_start)[0][0]

        if index_start in self.train_idx:
            dataset = self.TrainDataset
        elif index_start in self.val_idx:
            dataset = self.ValDataset
        elif index_start in self.test_idx:
            dataset = self.TestDataset
        else:
            raise ValueError('Invalid index')

        return dataset.get_data_from_ts(time, station)


class MetroDataset_deepar(MetroDataset_base):
    def __init__(self, data, args, f_index=None):
        super(MetroDataset_deepar, self).__init__(data, args, f_index)
        self.return_abnormal = False

    def __getitem__(self, index):
        data_piece = self.data.iloc[self.f_index[index]:self.f_index[index] + self.sample_len, :]
        outflow_data = torch.from_numpy(data_piece.outflow.values[:]).unfold(0, self.patch_len, self.patch_len)
        inflow_data = torch.from_numpy(data_piece.inflow.values[:]).unfold(0, self.patch_len, self.patch_len)
        target = torch.cat((inflow_data[1:, :], outflow_data[1:, :]), dim=1)
        input = torch.cat((inflow_data[:-1, :], outflow_data[:-1, :]), dim=1)

        inputs = [input]
        features = torch.from_numpy(data_piece[['station', 'weekday', 'time_in_day']].values[self.patch_len-1::self.patch_len])
        if self.station_eb:
            inputs.append(features[:-1, 0])
        if self.weekday_eb:
            inputs.append(features[:-1, 1])
        if self.time_eb:
            inputs.append(features[:-1, 2])

        if self.return_abnormal:  # Whether to return abnormal maker for the target, especially during testing
            abnormal_in = torch.from_numpy(data_piece.abnormal_in.values[:]).unfold(0, self.patch_len, self.patch_len)
            abnormal_out = torch.from_numpy(data_piece.abnormal_out.values[:]).unfold(0, self.patch_len, self.patch_len)
            abnormal = torch.cat((abnormal_in[1:, :], abnormal_out[1:, :]), dim=1)
            return tuple(inputs), target, abnormal
        else:
            return tuple(inputs), target

