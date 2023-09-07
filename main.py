import datetime

import pandas as pd
import pytz as pytz
import yfinance as yf

from model.preprocessors import process_inputs, process_targets
from model.helpers import train, predict
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

if __name__ == "__main__":
    # Download price histories from Yahoo Finance
    spy = yf.Ticker("SPY")
    price_series = spy.history(period='max')['Close'].dropna()

    # perf_series = price_series
    perf_series = price_series.pct_change().dropna()
    perf_series.to_pickle('./perf_series.pkl')

    perf_series = pd.read_pickle('./perf_series.pkl')

    x_df = process_inputs(perf_series, window_length=10)
    y_series = process_targets(perf_series)

    # Only keep rows in which we have both inputs and data.
    common_index = x_df.index.intersection(y_series.index)
    x_df, y_series = x_df.loc[common_index], y_series.loc[common_index]

    training_cutoff = datetime.datetime(2020, 1, 1, tzinfo=pytz.timezone('America/New_York'))
    training_x_series = x_df.loc[x_df.index < training_cutoff]
    training_y_series = y_series.loc[y_series.index < training_cutoff]

    trained_model = train(training_x_series, training_y_series)

    test_x_series = x_df.loc[x_df.index >= training_cutoff]
    forecast_series = predict(trained_model, test_x_series)
    actual_series = y_series.loc[y_series.index >= training_cutoff]
    results_df = forecast_series.to_frame('Forecast').join(actual_series.to_frame('Actual')).dropna()

    results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()
    print(f"R Squared: {r2_score(results_df['Actual'], results_df['Forecast']):.4f}, "
          f"Mean Absolute Error: {mean_absolute_error(results_df['Actual'], results_df['Forecast']):.4f}")
