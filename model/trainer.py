from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.dataset import LSTMStocksDataset
from model.model import LSTMStocksModule


def train(x_series: pd.Series, y_series: pd.Series, epochs: int = 100):
    # Put data into GPU if possible
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Store all training data in the GPU
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    x_tensor, y_tensor = torch.tensor(x_series.values), torch.tensor(y_series.values)

    train_dataset = LSTMStocksDataset(x_tensor, y_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = LSTMStocksModule(y_series.shape[1]).train()
    if torch.cuda.is_available():
        model = model.cuda()

    loss_func = partial(torch.nn.functional.huber_loss, delta=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    for i in range(epochs):
        total_loss = 0.
        for x, y in train_dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out, y)
            optimizer.step()
            total_loss += loss.cpu().detach().numpy()

        print(f"[Epoch {i}] Loss: {total_loss:.4f}")

    return model
