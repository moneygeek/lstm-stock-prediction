import datetime

import pandas as pd
import pytz as pytz
import yfinance as yf

from model.preprocessors import process_inputs, process_targets

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

    x = process_inputs(training_series, 10)
    y = process_targets(training_series)

    common_index = x.index.intersection(y.index)
    x, y = x.loc[common_index], y.loc[common_index]

    print(x)
