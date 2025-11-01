import opendatasets as od
from config import (
    RAW_DATA_PATH,
    INTERIM_DATA_PATH,
    dtypes,
    selected_cols,
    sample_fraction,
)
from pathlib import Path
import pandas as pd
import random


def download_dataset(url, path):
    save_path = Path(f"{path}")
    od.download(url, save_path)


def skip_row(row_idx):
    if row_idx == 0:
        return False
    return random.random() > sample_fraction


def load_csv(path, name="", **params):
    return pd.read_csv(f"{path}/{name}.csv", **params)


def save_parquet(df, path, name=""):
    save_path = Path(f"{path}/{name}.parquet")
    df.to_parquet(save_path)


if __name__ == "__main__":
    """
    download_dataset(
        "https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data",
        RAW_DATA_PATH,
    )
    """

    """
    Loading training data -
    As data is large we will use 1% of data
    ignoring key columns
    """

    random.seed(42)
    train_df = load_csv(
        RAW_DATA_PATH,
        name="train",
        usecols=selected_cols,
        dtype=dtypes,
        skiprows=skip_row,
        parse_dates=["pickup_datetime"],
    )

    test_df = load_csv(RAW_DATA_PATH, name="test", dtype=dtypes)
    save_parquet(train_df, INTERIM_DATA_PATH, name="100%_train_df")
    save_parquet(test_df, INTERIM_DATA_PATH, name="test_df")
