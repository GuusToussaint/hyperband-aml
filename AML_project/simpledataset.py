import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super(SimpleDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return torch.tensor(self.X[index, :], dtype=torch.float32), torch.tensor(int(self.y[index]), dtype=torch.long)
        # return self.X[index,:], int(self.y[index])

    def __len__(self):
        return self.X.shape[0]