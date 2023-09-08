from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.dataset import LSTMStocksDataset
from model.model import LSTMStocksModule
from matplotlib import pyplot as plt
from scipy.stats import norm


LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6


def _draw_chart(y_series: pd.Series):
    y_series.plot.hist(bins=50, label='Target Returns')

    # Draw Gaussian curve
    mu, stdev = norm.fit(y_series)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 50)
    p = norm.pdf(x, mu, stdev)
    p *= y_series.shape[0] / p.sum()

    plt.plot(x, p, 'k', linewidth=2, label='Gaussian Curve')
    plt.legend()


def chart_y_histogram(y_series: pd.Series):
    """
    Create charts that contain the histogram of the inputs and compares them against Normal distributions.
    :param y_series: Data to create charts for.
    """
    _draw_chart(y_series)
    plt.show()

    _draw_chart(y_series)
    xmin, _ = plt.xlim()
    plt.axis([xmin, -0.015, 0, 10])
    plt.show()


def train(x_series: pd.Series, y_series: pd.Series, epochs: int = 100):
    """
    Trains the LSTMStocksModule model
    :param x_series: Inputs consisting of sequences of stock price returns
    :param y_series: Targets consisting of returns some days in advance of the reference dates
    :param epochs: Number of complete iterations to go through the data in order to train
    :return: The trained LSTMStocksModule model
    """
    # Put data into GPU if possible
    dataloader_kwargs = {}
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Store all training data in the GPU
        dataloader_kwargs['generator'] = torch.Generator(device='cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Turn pandas objects into Pytorch tensor objects
    x_tensor, y_tensor = torch.tensor(x_series.values).float(), torch.tensor(y_series.values).float()

    # Set up the dataloader
    train_dataset = LSTMStocksDataset(x_tensor, y_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, **dataloader_kwargs)

    model = LSTMStocksModule().train()
    if torch.cuda.is_available():  # Train on GPU if possible
        model = model.cuda()

    loss_func = torch.nn.functional.cross_entropy
    chart_y_histogram(y_series)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Conduct training which consists of homing the model in on the best parameters that minimize the loss
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


def predict(trained_model, x_series: pd.Series) -> pd.Series:
    """
    Generates predictions using a trained model
    :param trained_model: Trained Pytorch model
    :param x_series: Inputs to generate predictions for
    :return: Series containing predictions, with reference dates as indices
    """
    trained_model.eval()

    x_tensor = torch.tensor(x_series.values).float()
    prediction = trained_model(x_tensor)

    return pd.Series(prediction.cpu().detach().numpy(), index=x_series.index)
