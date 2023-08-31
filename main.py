import datetime

import pandas as pd
import pytz as pytz
import yfinance as yf

from model.preprocessors import process_inputs, process_targets
from model.trainer import train

if __name__ == "__main__":
    # spy = yf.Ticker("SPY")
    # price_series = spy.history(period='10y')['Close']
    #
    # perf_series = price_series.pct_change().dropna()
    # perf_series.to_pickle('./perf_series.pkl')

    perf_series = pd.read_pickle('./perf_series.pkl')

    training_cutoff = datetime.datetime(2020, 1, 1, tzinfo=pytz.timezone('America/New_York'))
    training_series = perf_series.loc[perf_series.index < training_cutoff]
    test_series = perf_series.loc[perf_series.index >= training_cutoff]

    training_x_series = process_inputs(training_series, window_length=10)
    training_y_series = process_targets(training_series)

    common_index = training_x_series.index.intersection(training_y_series.index)
    training_x_series, training_y_series = training_x_series.loc[common_index], training_y_series.loc[common_index]

    trained_model = train(training_x_series, training_y_series)

    print(x)
