from torch import nn


class LSTMStocksModule(nn.Module):
    HIDDEN_SIZE = 1
    NUM_LAYERS = 1
    BIAS = True

    def __init__(self, window_length: int):
        super(LSTMStocksModule, self).__init__()
        self.lstm = nn.LSTM(
            1,
            self.HIDDEN_SIZE,
            self.NUM_LAYERS,
            self.BIAS,
            batch_first=True
        )
        if self.HIDDEN_SIZE > 1:
            self.linear = nn.Linear(self.HIDDEN_SIZE, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x.unsqueeze(-1))
        out = hidden.squeeze()
        if self.HIDDEN_SIZE > 1:
            out = self.linear(out).squeeze()
        return out

