# The new data utilities for metro datasets.
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
import warnings
import random
from models.basics import *

basic_infos = {
               'dropout' : 0.05,
               'attn_dropout' : 0.05,
               'head_dropout' : 0.05,
               'max_lr' : 0.001,
               'patience' : 5,
               'anneal_strategy' : 'linear',
               'batch_size' : 128,
               'd_model' : 128,
               'n_heads' : 8,
               'n_layers' : 3,
               'attn_mask_type' : 'time',
               'pre_norm' : True,
               'pe' : 'zeros',  # intial values of positional encoding, 'rotary'
               'learn_pe' : True,  # learn positional encoding
               'div_factor' : 1e4,  # initial warmup learning rate : max_lr / div_factor
               'final_div_factor' : 1,  # final learning rate : initial_lr / final_div_factor
               'input_emb_size' : 8,
               'max_leap_rate' : 0.1,
               'initial_num_bins' : 1024,
               'train_method' : None,
}

data_infos = {'guangzhou': {'data_path': '../../data/GuangzhouMetro//',
                            'inflow_file': 'in_data.csv',
                            'outflow_file': 'out_data.csv',
                            'start_minute': 360,
                            'end_minute': 1440,
                            't_resolution': '15T',  # time resolution used in the model
                            'input_len': 144,
                            'target_len': 48,
                            'test_date_start': '2017-09-15',  # equivalent to 2017-09-16
                            'test_date_end': '2017-10-01',
                            'val_date_start': '2017-09-08',  # equivalent to 2017-09-09
                            'val_date_end': '2017-09-15',
                            'sample_interval':1,  # The skip interval for fast testing
                            'sample_divide':4,
                            'patch_len':1
                            },
              'seoul': {'data_path': '../../data/SeoulMetro/',
                        'inflow_file': 'in_data.csv',
                        'outflow_file': 'out_data.csv',
                        'start_minute': 240,
                        'end_minute': 1381,
                        't_resolution': '60T',  # time resolution used in the model
                        'input_len': 40,
                        'target_len': 12,
                        'test_date_start': '2023-06-09',  # equivalent to 2023-06-10
                        'test_date_end': '2023-07-01',
                        'val_date_start': '2023-05-31',  # equivalent to 2023-06-01
                        'val_date_end': '2023-06-09',
                        'sample_interval': 1,  # The skip interval for fast testing
                        'patch_len':1,
                        'sample_divide':4,
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
                              'dropout': 0,
                              'attn_dropout': 0.05,
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
                       },
               'ABT2': {'d_model': 128,
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
                       },
               'ABT_new': {'d_model': 128,
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
                        },
               'Nlinear':{
                   'station_eb': True,
                   'weekday_eb': True,
                   'time_eb': True,
                   'n_epochs': 20,
                   'patience': 5,
                   'max_lr': 0.001,
                   'seed': 0,
                   'max_run_time': 10,
                   'd_model': 128,
                   'dropout': 0.05,
               }
                }

head_infos = {'RMSE': {'standardization' : 'zscore', 'output_type': 'number', 'input_type': 'number'},
              'NB': {'standardization' : 'meanscale', 'output_type': 'number', 'input_type': 'number'},
              'logNormal': {'standardization' : 'zscore', 'output_type': 'number', 'input_type': 'number'},
              'CrossEntropy': {'standardization' : 'none', 'output_type': 'bins', 'top_p': 0.98, 'input_type': 'bins'},
              'TruncatedNormal': {'standardization' : 'zscore', 'output_type': 'number', 'input_type': 'number'},
              'Normal': {'standardization' : 'zscore', 'output_type': 'number', 'input_type': 'number'},
              'MixTruncatedNormal': {'standardization' : 'zscore', 'output_type': 'number', 'input_type': 'number', 'n_mixture': 2},
              'MixTruncatedNormal2': {'standardization' : 'zscore', 'output_type': 'number', 'input_type': 'number', 'n_mixture': 2},
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
        self.test_mode = False

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
        test_mode = self.test_mode
        self.test_mode = True
        index_now = self.data[(self.data.time == time) & (self.data.station == station)].index.values[0]
        index_start = index_now - self.input_len
        if index_start not in self._index:
            print('Invalid or infeasible time and station.')
            return None

        data_piece = self.data.iloc[index_start:index_now + self.target_len, :]

        # Get the location of index in self._index.
        index = np.where(self._index == index_start)[0][0]
        x, y, abnormal = self.__getitem__(index)
        self.test_mode = test_mode
        return (x, y, abnormal), data_piece


class MetroDataset_total(MetroDataset_base):
    def __init__(self, data, args, f_index=None):
        data.sort_values(['station', 'time'],inplace=True)
        data.reset_index(drop=True, inplace=True)
        self.raw_data = data
        self.train_idx, self.val_idx, self.test_idx = get_train_val_test_index(self.raw_data, args)
        self.bin_data = data.copy()
        self.input_type = args.input_type  # 'number' or "bins"
        self.output_type = args.output_type  # 'number' or "bins"
        self.forecast_target = args.forecast_target

        if args.output_type == 'bins' or args.input_type == 'bins':
            self.bin_edges, self.num_per_bin, _ = get_quantile_edges(self.bin_data.loc[self.train_idx, ['inflow', 'outflow']].values.flatten(),
                                                args.initial_num_bins, args.max_leap_rate, plot=False)
            self.bin_data['outflow'] = self.bin_data.outflow.apply(lambda x: num2bin(x, self.bin_edges))
            self.bin_data['inflow'] = self.bin_data.inflow.apply(lambda x: num2bin(x, self.bin_edges))
            self.bin_data['outflow'] = self.bin_data.outflow.astype('int64')
            self.bin_data['inflow'] = self.bin_data.inflow.astype('int64')
            self.bin_weights = self.num_per_bin.sum()/(self.num_per_bin * len(self.num_per_bin))
        else:
            self.bin_edges = np.array([0])
            self.bin_data = None
            self.bin_weights = 1
        self.num_bins = len(self.bin_edges) - 1

        super(MetroDataset_total, self).__init__(self.raw_data, args, f_index)

        datatype_dict = {'ABT': MetroDataset_deepar,
                         'ABT2': MetroDataset_deepar,
                         'DeepAR': MetroDataset_deepar,
                         'ABT_concat': MetroDataset_deepar,  # ABT_concat uses the same data structure as DeepAR
                         'ABT_new': MetroDataset_deepar,
                         'Nlinear': MetroDataset_seq2seq,
                         }
        child_dataset = datatype_dict[args.model]

        # Get train, val, test Dataset
        self.TrainDataset = child_dataset(self.raw_data, self.bin_data, args,
                                          np.intersect1d(self.train_idx, self.f_index), forecast_target=args.forecast_target)
        self.ValDataset = child_dataset(self.raw_data, self.bin_data, args,
                                        np.intersect1d(self.val_idx, self.f_index), forecast_target=args.forecast_target)
        self.TestDataset = child_dataset(self.raw_data, self.bin_data, args,
                                         np.intersect1d(self.test_idx, self.f_index), forecast_target=args.forecast_target)

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
    def __init__(self, raw_data, bin_data, args, f_index=None, forecast_target='both'):
        super(MetroDataset_deepar, self).__init__(raw_data, args, f_index)
        self.raw_data = raw_data
        self.bin_data = bin_data
        self.input_type = args.input_type  # 'number' or "bins"
        self.output_type = args.output_type  # 'number' or "bins"
        self.forecast_target = forecast_target

    def __getitem__(self, index):
        # the input data
        if self.input_type == 'bins':
            bin_data_piece = self.bin_data.iloc[self.f_index[index]:self.f_index[index] + self.sample_len, :]
            bin_inflow = torch.from_numpy(bin_data_piece.inflow.values[:]).unfold(0, self.patch_len, self.patch_len)
            bin_outflow = torch.from_numpy(bin_data_piece.outflow.values[:]).unfold(0, self.patch_len, self.patch_len)
            if self.forecast_target == 'inflow':
                input = bin_inflow[:-1, :]
            elif self.forecast_target == 'outflow':
                input = bin_outflow[:-1, :]
            else:
                input = torch.cat((bin_inflow[:-1, :], bin_outflow[:-1, :]), dim=1)
        elif self.input_type == 'number':
            data_piece = self.raw_data.iloc[self.f_index[index]:self.f_index[index] + self.sample_len, :]
            inflow = torch.from_numpy(data_piece.inflow.values[:]).unfold(0, self.patch_len, self.patch_len)
            outflow = torch.from_numpy(data_piece.outflow.values[:]).unfold(0, self.patch_len, self.patch_len)
            if self.forecast_target == 'inflow':
                input = inflow[:-1, :]
            elif self.forecast_target == 'outflow':
                input = outflow[:-1, :]
            else:
                input = torch.cat((inflow[:-1, :], outflow[:-1, :]), dim=1)
        else:
            raise ValueError('Invalid input type. input_type should be "number" or "bins"')

        # Get features
        inputs = [input]
        data = data_piece if 'data_piece' in locals() else bin_data_piece
        features = torch.from_numpy(data[['station', 'weekday', 'time_in_day']].values[self.patch_len-1::self.patch_len])
        if self.station_eb:
            inputs.append(features[:-1, 0])
        if self.weekday_eb:
            inputs.append(features[:-1, 1])
        if self.time_eb:
            inputs.append(features[:-1, 2])

        # Get output data
        if self.output_type == 'bins' and not self.test_mode:
            if "bin_data_piece" not in locals():
                bin_data_piece = self.bin_data.iloc[self.f_index[index]:self.f_index[index] + self.sample_len, :]
                bin_inflow = torch.from_numpy(bin_data_piece.inflow.values[:]).unfold(0, self.patch_len, self.patch_len)
                bin_outflow = torch.from_numpy(bin_data_piece.outflow.values[:]).unfold(0, self.patch_len, self.patch_len)
            if self.forecast_target == 'inflow':
                target = bin_inflow[1:, :]
            elif self.forecast_target == 'outflow':
                target = bin_outflow[1:, :]
            else:
                target = torch.cat((bin_inflow[1:, :], bin_outflow[1:, :]), dim=1)
        elif self.output_type == 'number' or self.test_mode:
            if "data_piece" not in locals():
                data_piece = self.raw_data.iloc[self.f_index[index]:self.f_index[index] + self.sample_len, :]
                inflow = torch.from_numpy(data_piece.inflow.values[:]).unfold(0, self.patch_len, self.patch_len)
                outflow = torch.from_numpy(data_piece.outflow.values[:]).unfold(0, self.patch_len, self.patch_len)
            if self.forecast_target == 'inflow':
                target = inflow[1:, :]
            elif self.forecast_target == 'outflow':
                target = outflow[1:, :]
            else:
                target = torch.cat((inflow[1:, :], outflow[1:, :]), dim=1)
        else:
            raise ValueError('Invalid output type. output_type should be "number" or "bins"')

        # Whether to return abnormal maker for the target, which is needed during testing
        if self.test_mode:
            data = data_piece if 'data_piece' in locals() else bin_data_piece
            abnormal_in = torch.from_numpy(data.abnormal_in.values[:]).unfold(0, self.patch_len, self.patch_len)
            abnormal_out = torch.from_numpy(data.abnormal_out.values[:]).unfold(0, self.patch_len, self.patch_len)
            if self.forecast_target == 'inflow':
                abnormal = abnormal_in[1:, :]
            elif self.forecast_target == 'outflow':
                abnormal = abnormal_out[1:, :]
            else:
                abnormal = torch.cat((abnormal_in[1:, :], abnormal_out[1:, :]), dim=1)
            return tuple(inputs), target, abnormal
        else:
            return tuple(inputs), target

class MetroDataset_seq2seq(MetroDataset_base):
    # The dataset designed for model that forecast multistep feature values jointly.
    def __init__(self, raw_data, bin_data, args, f_index=None, forecast_target='both'):
        super(MetroDataset_seq2seq, self).__init__(raw_data, args, f_index)
        self.raw_data = raw_data
        self.forecast_target = forecast_target

    def __getitem__(self, index):
        # the input data
        data_piece = self.raw_data.iloc[self.f_index[index]:self.f_index[index] + self.sample_len, :]
        inflow = torch.from_numpy(data_piece.inflow.values[:]).unfold(0, self.patch_len, self.patch_len)
        outflow = torch.from_numpy(data_piece.outflow.values[:]).unfold(0, self.patch_len, self.patch_len)
        if self.forecast_target == 'inflow':
            input = inflow[:self.input_len, :]
            target = inflow[-self.target_len:, :]
        elif self.forecast_target == 'outflow':
            input = outflow[:self.input_len, :]
            target = outflow[-self.target_len:, :]
        else:
            input = torch.cat((inflow[:self.input_len, :], outflow[:self.input_len, :]), dim=1)
            target = torch.cat((inflow[-self.target_len:, :], outflow[-self.target_len:, :]), dim=1)

        # Get features
        inputs = [input]
        data = data_piece
        features = torch.from_numpy(data[['station', 'weekday', 'time_in_day']].values[self.patch_len-1::self.patch_len])
        if self.station_eb:
            inputs.append(features[-self.target_len:, 0])
        if self.weekday_eb:
            inputs.append(features[-self.target_len:, 1])
        if self.time_eb:
            inputs.append(features[-self.target_len:, 2])

        if self.test_mode:
            data = data_piece
            abnormal_in = torch.from_numpy(data.abnormal_in.values[:]).unfold(0, self.patch_len, self.patch_len)
            abnormal_out = torch.from_numpy(data.abnormal_out.values[:]).unfold(0, self.patch_len, self.patch_len)
            if self.forecast_target == 'inflow':
                abnormal = abnormal_in[-self.target_len:, :]
            elif self.forecast_target == 'outflow':
                abnormal = abnormal_out[-self.target_len:, :]
            else:
                abnormal = torch.cat((abnormal_in[-self.target_len:, :], abnormal_out[-self.target_len:, :]), dim=1)
            return tuple(inputs), target, abnormal
        else:
            return tuple(inputs), target



