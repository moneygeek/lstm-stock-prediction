import pandas as pd


def process_inputs(perf_series: pd.Series, window_length: int):
    dataframes = []
    for i in range(window_length):
        dataframes.append(perf_series.shift(i).to_frame(f"T - {i}"))

    return pd.concat(reversed(dataframes), axis=1).dropna()


def process_targets(perf_series: pd.Series):
    return perf_series.shift(-2).dropna()
