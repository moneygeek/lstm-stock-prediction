import datetime

import pandas as pd
import pytz as pytz
import yfinance as yf

from model.preprocessors import process_inputs, process_targets
from model.helpers import train, predict
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

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

    test_x_series = process_inputs(test_series, window_length=10)
    forecast_series = predict(trained_model, test_x_series)
    actual_series = process_targets(test_series)
    results_df = forecast_series.to_frame('Forecast').join(actual_series.to_frame('Actual')).dropna()

    results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()
    print(f"R Squared: {r2_score(results_df['Actual'], results_df['Forecast']):.4f}, "
          f"Mean Absolute Error: {mean_absolute_error(results_df['Actual'], results_df['Forecast']):.4f}")
