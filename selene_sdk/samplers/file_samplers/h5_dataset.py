import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class H5Dataset(data.Dataset):
    def __init__(self, file_path, in_memory=False):
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        self.db_len = None
        self.initialized = False
        self.in_memory = in_memory

    def init(func):
        #delay initlization to allow  multiprocessing
        def dfunc(self, *args, **kwargs):
            if not self.initialized:
                self.db = h5py.File(self.file_path, 'r')
                self.s_len = self.db['sequences_length'][()]
                self.t_len = self.db['targets_length'][()]
                if self.in_memory:
                    self.sequences = np.asarray(self.db["sequences"])
                    self.targets = np.asarray(self.db["targets"])
                else:
                    self.sequences = self.db["sequences"]
                    self.targets = self.db["targets"]
                self.initialized = True
            return func(self, *args, **kwargs)
        return dfunc

    @init
    def __getitem__(self, index):
        sequence = np.unpackbits(self.sequences[index, :, :],axis=-2)
        nulls = np.sum(sequence, axis=-1) == 4
        sequence = sequence.astype(float)
        sequence[nulls, :] = 0.25
        targets = np.unpackbits(self.targets[index, :],axis=-1).astype(float)
        if sequence.ndim == 3:
            sequence = sequence[:,:self.s_len,:]
        else:
            sequence = sequence[:self.s_len,:]
        if targets.ndim == 2:
            targets = targets[:,:self.t_len]
        else:
            targets = targets[:self.t_len]
        return (torch.from_numpy(sequence).float(), torch.from_numpy(targets).float())

    @init
    def __len__(self):
        if self.db_len:
            return self.db_len

        self.db_len = self.sequences.shape[0]
        return self.db_len


class H5DataLoader(DataLoader):
    def __init__(self,
                 filepath,
                 in_memory=False,
                 num_workers=1,
                 use_subset=None,
                 batch_size=1,
                 shuffle=True):
         args = {
             "batch_size": batch_size,
             "num_workers": 0 if in_memory else num_workers,
             "pin_memory": True
         }
         if use_subset is not None:
             from torch.utils.data.sampler import SubsetRandomSampler
             if type(use_subset, int):
                 use_subset = list(range(use_subset))
             args["sampler"] = SubsetRandomSampler(use_subset)
         else:
             args["shuffle"] = shuffle
         super(H5DataLoader, self).__init__(H5Dataset(filepath, in_memory=in_memory),**args)

    def get_data_and_targets(self, batch_size, n_samples=None):
        sequences, targets = self.dataset[:n_samples]
        return sequences.numpy(),targets.numpy()


