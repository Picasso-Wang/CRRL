import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
import numpy as np
import pickle
import random


class SelfDefineDataset_pretrain_speed(Dataset):
    def __init__(self, data_path, data_reverse_path, num_frame_path, speed_path, speed_reverse_path, dataset='ntu'):
        if dataset == 'ntu' or dataset == 'ntu120':
            self.data = np.load(data_path)
            self.data_reverse = np.load(data_reverse_path)
            self.num_frame = np.load(num_frame_path)
            self.speed = np.load(speed_path)
            self.speed_reverse = np.load(speed_reverse_path)

        elif dataset == 'cmu' or dataset == 'cmu_subset' or dataset == 'ucla' or dataset == 'pku1' or dataset == 'pku2':
            self.data = np.load(data_path, allow_pickle=True)
            self.data_reverse = np.load(data_reverse_path, allow_pickle=True)
            self.num_frame = np.load(num_frame_path)
            self.speed = np.load(speed_path, allow_pickle=True)
            self.speed_reverse = np.load(speed_reverse_path, allow_pickle=True)

    def __len__(self):
        return len(self.num_frame)

    def __getitem__(self, idx):
        number_frame = self.num_frame[idx]
        data_numpy = self.data[idx]
        data_rev_numpy = self.data_reverse[idx]
        speed_numpy = self.speed[idx]
        speed_rev_numpy = self.speed_reverse[idx]
        return number_frame, torch.from_numpy(data_numpy), torch.from_numpy(data_rev_numpy), torch.from_numpy(speed_numpy), torch.from_numpy(speed_rev_numpy)


def collate_fn_speed(data_list):
    data_list.sort(key=lambda x: x[0], reverse=True)
    num_frame = [a[0] for a in data_list]
    data_x = [a[1] for a in data_list]
    data_y = [a[2] for a in data_list]
    data_s = [a[3] for a in data_list]
    data_sr = [a[4] for a in data_list]

    data_x = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0.0)
    data_y = rnn_utils.pad_sequence(data_y, batch_first=True, padding_value=0.0)
    data_s = rnn_utils.pad_sequence(data_s, batch_first=True, padding_value=0.0)
    data_sr = rnn_utils.pad_sequence(data_sr, batch_first=True, padding_value=0.0)

    return num_frame, data_x, data_y, data_s, data_sr


class SelfDefineDataset_linEval(Dataset):
    def __init__(self, data_path, num_frame_path, label_path, dataset='ntu'):
        if dataset == 'ntu' or dataset == 'ntu120':
            self.data = np.load(data_path)
            self.num_frame = np.load(num_frame_path)
            self.label = np.load(label_path)

        elif dataset == 'cmu' or dataset == 'cmu_subset' or dataset == 'ucla' or dataset == 'pku1' or dataset == 'pku2':
            self.data = np.load(data_path, allow_pickle=True)
            self.num_frame = np.load(num_frame_path)
            self.label = (np.load(label_path)).astype(np.int64)

    def __len__(self):
        return len(self.num_frame)

    def __getitem__(self, idx):
        number_frame = self.num_frame[idx]
        data_numpy = self.data[idx]
        label_ = self.label[idx]
        return number_frame, torch.from_numpy(data_numpy), label_


def collate_fn_linEval(data_list):
    data_list.sort(key=lambda x: x[0], reverse=True)
    num_frame = [a[0] for a in data_list]
    data = [a[1] for a in data_list]
    label = [a[2] for a in data_list]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)

    return num_frame, data, torch.tensor(label)
