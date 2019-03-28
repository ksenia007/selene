import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class H5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        self.db_len = None
        with h5py.File(self.file_path, 'r') as db:
            self.s_len = db['sequences_length'][()]
            self.t_len = db['targets_length'][()]

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as db:
            sequence = np.unpackbits(db["sequences"][index, :, :],axis=-2)
            nulls = np.sum(sequence, axis=-1) == 4
            sequence = sequence.astype(float)
            sequence[nulls, :] = 0.25
            targets = np.unpackbits(db["targets"][index, :],axis=-1).astype(float)
            if sequence.ndim == 3:
                sequence = sequence[:,:self.s_len,:]
            else:
                sequence = sequence[:self.s_len,:]
            if targets.ndim == 2:
                targets = targets[:,:self.t_len]
            else:
                targets = targets[:self.t_len]
            return (torch.from_numpy(sequence).float(), torch.from_numpy(targets).float())

    def __len__(self):
        if self.db_len:
            return self.db_len

        with h5py.File(self.file_path, 'r') as db:
            self.db_len = db["sequences"].shape[0]
            return self.db_len


class H5DataLoader(DataLoader):
    def __init__(self, filepath, num_workers=1, use_subset=None, batch_size=1,shuffle=True):
         args = {
             "batch_size": batch_size,
             "num_workers": num_workers,
             "pin_memory": True
         }
         if use_subset is not None:
             from torch.utils.data.sampler import SubsetRandomSampler
             if type(use_subset, int):
                 use_subset = list(range(use_subset))
             args["sampler"] = SubsetRandomSampler(use_subset)
         else:
             args["shuffle"] = shuffle
         super(H5DataLoader, self).__init__(H5Dataset(filepath),**args)

    def get_data_and_targets(self, batch_size, n_samples=None):
        return self.dataset[:n_samples]


def h5_dataloader(filepath,
                  num_workers=1,
                  use_subset=None,
                  batch_size=1,
                  shuffle=True):
    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True
    }
    if use_subset is not None:
        from torch.utils.data.sampler import SubsetRandomSampler
        if type(use_subset, int):
            use_subset = list(range(use_subset))
        args["sampler"] = SubsetRandomSampler(use_subset)
    else:
        args["shuffle"] = shuffle
    return DataLoader(
        H5Dataset(filepath),
        **args)

