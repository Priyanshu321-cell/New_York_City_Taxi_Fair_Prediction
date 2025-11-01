from config import INTERIM_DATA_PATH, input_cols, target_cols, PROCESSED_DATA_PATH
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from dataset import save_parquet
import numpy as np


def load_parquet(path, name=""):
    save_path = Path(f"{path}/{name}.parquet")
    return pd.read_parquet(save_path)


def add_dateparts(df, col):
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month
    df[col + "_day"] = df[col].dt.day
    df[col + "_weekday"] = df[col].dt.weekday
    df[col + "_hour"] = df[col].dt.hour


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat2, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def add_trip_distance(df):
    df["trip_distance"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )


def add_landmark_drop_off_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + "_drop_distance"] = haversine_np(
        lon, lat, df["dropoff_longitude"], df["dropoff_latitude"]
    )


def add_landmark_pick_up_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + "_pick_distance"] = haversine_np(
        lon, lat, df["pickup_longitude"], df["pickup_latitude"]
    )


def remove_outliers(df):
    return df[
        (df["fare_amount"] >= 1.0)
        & (df["fare_amount"] < 500.0)
        & (df["pickup_longitude"] >= -75)
        & (df["pickup_longitude"] <= -72)
        & (df["dropoff_longitude"] >= -75)
        & (df["dropoff_longitude"] <= -72)
        & (df["pickup_latitude"] >= 40)
        & (df["pickup_latitude"] <= 42)
        & (df["dropoff_latitude"] >= 40)
        & (df["dropoff_latitude"] <= 42)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
    ]


if __name__ == "__main__":
    df = load_parquet(INTERIM_DATA_PATH, name="100%_train_df")
    test_df = load_parquet(INTERIM_DATA_PATH, name="test_df")

    # extractind dates
    df["pickup_datetime"] = pd.to_datetime(df.pickup_datetime)
    test_df["pickup_datetime"] = pd.to_datetime(test_df.pickup_datetime)

    add_dateparts(df, "pickup_datetime")
    add_dateparts(test_df, "pickup_datetime")

    # train test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    len(train_df), len(val_df)

    # fill and remove missing
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    train_df.columns

    # adding trip diistance
    add_trip_distance(train_df)
    add_trip_distance(val_df)
    add_trip_distance(test_df)

    # adding distance from popular landmarks
    jfk_lonlat = -73.7781, 40.6413
    lga_lonlat = -73.8740, 40.7769
    ewr_lonlat = -74.1745, 40.6895
    met_lonlat = -73.9632, 40.7794
    ts_lonlat = 40.7588, -73.9851
    wtc_lonlat = -74.0099, 40.7126

    add_landmark_drop_off_distance(train_df, "JFK_airport", jfk_lonlat)
    add_landmark_pick_up_distance(train_df, "JFK_airport", jfk_lonlat)
    add_landmark_drop_off_distance(train_df, "LGA_airport", lga_lonlat)
    add_landmark_pick_up_distance(train_df, "LGA_airport", lga_lonlat)
    add_landmark_drop_off_distance(train_df, "EWR_airport", ewr_lonlat)
    add_landmark_pick_up_distance(train_df, "EWR_airport", ewr_lonlat)
    add_landmark_drop_off_distance(train_df, "Times_square", ts_lonlat)
    add_landmark_pick_up_distance(train_df, "Times_square", ts_lonlat)
    add_landmark_drop_off_distance(train_df, "Met_meuseum", met_lonlat)
    add_landmark_pick_up_distance(train_df, "Met_meuseum", met_lonlat)
    add_landmark_drop_off_distance(train_df, "World_trade_center", wtc_lonlat)
    add_landmark_pick_up_distance(train_df, "World_trade_center", wtc_lonlat)

    add_landmark_drop_off_distance(val_df, "JFK_airport", jfk_lonlat)
    add_landmark_pick_up_distance(val_df, "JFK_airport", jfk_lonlat)
    add_landmark_drop_off_distance(val_df, "LGA_airport", lga_lonlat)
    add_landmark_pick_up_distance(val_df, "LGA_airport", lga_lonlat)
    add_landmark_drop_off_distance(val_df, "EWR_airport", ewr_lonlat)
    add_landmark_pick_up_distance(val_df, "EWR_airport", ewr_lonlat)
    add_landmark_drop_off_distance(val_df, "Times_square", ts_lonlat)
    add_landmark_pick_up_distance(val_df, "Times_square", ts_lonlat)
    add_landmark_drop_off_distance(val_df, "Met_meuseum", met_lonlat)
    add_landmark_pick_up_distance(val_df, "Met_meuseum", met_lonlat)
    add_landmark_drop_off_distance(val_df, "World_trade_center", wtc_lonlat)
    add_landmark_pick_up_distance(val_df, "World_trade_center", wtc_lonlat)

    add_landmark_drop_off_distance(test_df, "JFK_airport", jfk_lonlat)
    add_landmark_pick_up_distance(test_df, "JFK_airport", jfk_lonlat)
    add_landmark_drop_off_distance(test_df, "LGA_airport", lga_lonlat)
    add_landmark_pick_up_distance(test_df, "LGA_airport", lga_lonlat)
    add_landmark_drop_off_distance(test_df, "EWR_airport", ewr_lonlat)
    add_landmark_pick_up_distance(test_df, "EWR_airport", ewr_lonlat)
    add_landmark_drop_off_distance(test_df, "Times_square", ts_lonlat)
    add_landmark_pick_up_distance(test_df, "Times_square", ts_lonlat)
    add_landmark_drop_off_distance(test_df, "Met_meuseum", met_lonlat)
    add_landmark_pick_up_distance(test_df, "Met_meuseum", met_lonlat)
    add_landmark_drop_off_distance(test_df, "World_trade_center", wtc_lonlat)
    add_landmark_pick_up_distance(test_df, "World_trade_center", wtc_lonlat)

    # remove outliers and invalid data
    """limiting following
    fare_amount: $1 to $ 500
    longitudes: -75 to -72
    latitudes : 40 to 42
    passenger_count : 1 to 6
    """
    train_df = remove_outliers(train_df)
    val_df = remove_outliers(val_df)

    # Scaling and one hot encoders
    """passed"""

    # inputs and outputs
    train_inputs = train_df[input_cols]
    train_targets = train_df[target_cols]
    val_inputs = val_df[input_cols]
    val_targets = val_df[target_cols]
    test_inputs = test_df[input_cols]

    # saving
    save_parquet(train_inputs, PROCESSED_DATA_PATH, name="train_inputs")
    save_parquet(train_targets.to_frame(), PROCESSED_DATA_PATH, name="train_targets")
    save_parquet(val_inputs, PROCESSED_DATA_PATH, name="val_inputs")
    save_parquet(val_targets.to_frame(), PROCESSED_DATA_PATH, name="val_targets")
    save_parquet(test_inputs, PROCESSED_DATA_PATH, name="test_inputs")
