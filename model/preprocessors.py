import pandas as pd


def process_inputs(perf_series: pd.Series, window_length: int) -> pd.DataFrame:
    """
    Creates sequences consisting of data from across a moving window. For example, given a window length of 10,
    sequences will span indices 0-9, 1-10, 2-11, etc.
    :param perf_series: The stock price returns data to extract sequences from.
    :param window_length: The size of the moving window.
    :return: Pandas DataFrame where each row contains a sequence, and the index refers to the most recent input date,
    a.k.a. the reference date.
    """
    dataframes = []
    for i in range(window_length):
        dataframes.append(perf_series.shift(i).to_frame(f"T - {i}"))

    return pd.concat(reversed(dataframes), axis=1).dropna()


def process_targets(perf_series: pd.Series) -> pd.Series:
    """
    Creates targets consisting of data 2 days after the reference date (i.e. the most recent input's date)
    :param perf_series: The stock price returns data to extract targets from.
    :return: A series where the values consist of returns 2 days after the reference dates, and where the index consists
    of the reference dates.
    """
    return perf_series.shift(-2).dropna()
