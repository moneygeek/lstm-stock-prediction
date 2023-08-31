from functools import partial

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.dataset import LSTMStocksDataset
from model.model import LSTMStocksModule


def train(x_series: pd.Series, y_series: pd.Series, epochs: int = 100):
    # Put data into GPU if possible
    dataloader_kwargs = {}
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Store all training data in the GPU
        dataloader_kwargs['generator'] = torch.Generator(device='cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    x_tensor, y_tensor = torch.tensor(x_series.values).float(), torch.tensor(y_series.values).float()

    train_dataset = LSTMStocksDataset(x_tensor, y_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, **dataloader_kwargs)

    model = LSTMStocksModule(x_series.shape[1]).train()
    if torch.cuda.is_available():
        model = model.cuda()

    loss_func = partial(torch.nn.functional.huber_loss, delta=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    for i in range(epochs):
        total_loss = 0.
        for x, y in train_dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().numpy()

        print(f"[Epoch {i}] Loss: {total_loss:.4f}")

    return model


def predict(trained_model, x_series: pd.Series):
    trained_model.eval()

    x_tensor = torch.tensor(x_series.values).float()
    prediction = trained_model(x_tensor)

    return pd.Series(prediction.cpu().detach().numpy(), index=x_series.index)
