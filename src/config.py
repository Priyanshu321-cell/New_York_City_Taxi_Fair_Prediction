MODEL_PATH = "../models"
RAW_DATA_PATH = "../data/raw/new-york-city-taxi-fare-prediction"
INTERIM_DATA_PATH = "../data/interim"
PROCESSED_DATA_PATH = "../data/processed"
selected_cols = list(
    "fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count".split(
        sep=","
    )
)
dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
sample_fraction = 0.01
input_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetime_year",
    "pickup_datetime_month",
    "pickup_datetime_day",
    "pickup_datetime_weekday",
    "pickup_datetime_hour",
    "trip_distance",
    "JFK_airport_drop_distance",
    "JFK_airport_pick_distance",
    "LGA_airport_drop_distance",
    "LGA_airport_pick_distance",
    "EWR_airport_drop_distance",
    "EWR_airport_pick_distance",
    "Times_square_drop_distance",
    "Times_square_pick_distance",
    "Met_meuseum_drop_distance",
    "Met_meuseum_pick_distance",
    "World_trade_center_drop_distance",
    "World_trade_center_pick_distance",
]
target_cols = "fare_amount"
