import h5py
import torch
import torch.utils.data as data


class H5Dataset(data.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        self.db_len = None

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as db:
            return (
                torch.from_numpy(db["sequences"][index, :, :].astype(float)).float(),
                torch.from_numpy(db["targets"][index, :].astype(float)).float())

    def __len__(self):
        if self.db_len:
            return self.db_len

        with h5py.File(self.file_path, 'r') as db:
            self.db_len = db["sequences"].shape[0]
            return self.db_len
