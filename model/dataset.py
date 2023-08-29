import torch
from torch.utils.data import Dataset


class LSTMStocksDataset(Dataset):
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self._x_tensor = x_tensor
        self._y_tensor = y_tensor

    def __len__(self):
        return self._y_tensor.shape[0]

    def __getitem__(self, idx: int):
        return self._x_tensor[idx, :], self._y_tensor[idx]
