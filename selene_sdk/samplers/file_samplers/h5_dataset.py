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

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as db:
            sequence = np.unpackbits(db["sequences"][index, :, :],axis=-2)
            nulls = np.sum(sequence, axis=-1) == 4
            sequence = sequence.astype(float)
            sequence[nulls, :] = 0.25
            targets = np.unpackbits(db["targets"][index, :],axis=-1).astype(float)
            return (torch.from_numpy(sequence).float(), torch.from_numpy(targets).float())

    def __len__(self):
        if self.db_len:
            return self.db_len

        with h5py.File(self.file_path, 'r') as db:
            self.db_len = db["sequences"].shape[0]
            return self.db_len


def h5_dataloader(filepath,
                  num_workers=1,
                  use_subset=None,
                  batch_size=1):
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
        args["shuffle"] = True
    return DataLoader(
        H5Dataset(filepath),
        **args)

