import h5py
import numpy as np
from pathlib import Path
import torch
import random
from torch.utils import data


class HDF5Dataset(data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h_file = h5py.File(h5_path, 'r')
        self.length = len(self.h_file['dataset']['id'])

    def __getitem__(self, index):
        data = self.h_file['dataset']['data'][index]
        label = self.h_file['dataset']['label'][index]
        sound_id = self.h_file['dataset']['id'][index]
        return data, label, sound_id

    def __len__(self):
        return self.length


class InMemoryDataset(data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as h_file:
            self.data = h_file['dataset']['data'][:]
            self.cf_data = h_file['dataset']['cf_data'][:]
            self.label = h_file['dataset']['label'][:]
            self.id = h_file['dataset']['id'][:]
            self.length = len(self.id)

    def __getitem__(self, index):
        rnd_index = np.random.randint(0,1300-256)
        return self.data[index][rnd_index:(rnd_index+256)], self.label[index], self.cf_data[index], self.id[index]

    def __len__(self):
        return self.length

