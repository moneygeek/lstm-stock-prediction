from torch import nn


class LSTMStocksModule(nn.Module):
    HIDDEN_SIZE = 2
    NUM_LAYERS = 1
    BIAS = True

    def __init__(self, window_length: int):
        super(LSTMStocksModule, self).__init__()
        self.lstm = nn.LSTM(
            window_length,
            self.HIDDEN_SIZE,
            self.NUM_LAYERS,
            self.BIAS
        )

    def forward(self, x):
        _, output = self.lstm(x)
        return output
