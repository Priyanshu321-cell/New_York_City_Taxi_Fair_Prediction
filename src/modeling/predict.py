from xgboost import XGBRegressor
from modeling.train import evaluate_model, predict_and_submit
from features import load_parquet
from config import PROCESSED_DATA_PATH, MODEL_PATH
import joblib


def dump_model(model, path, name=""):
    joblib.dump(model, f"{path}/{name}.joblib")


def load_model(path, name=""):
    joblib.load(f"{path}/{name}.joblib")


if __name__ == "__main__":
    train_inputs = load_parquet(PROCESSED_DATA_PATH, name="train_inputs")
    train_targets = load_parquet(PROCESSED_DATA_PATH, name="train_targets")
    val_inputs = load_parquet(PROCESSED_DATA_PATH, name="val_inputs")
    val_targets = load_parquet(PROCESSED_DATA_PATH, name="val_targets")
    test_inputs = load_parquet(PROCESSED_DATA_PATH, name="test_inputs")

    model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        learning_rate=0.08,
        n_estimators=5000,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
    ).fit(train_inputs, train_targets)

    evaluate_model(
        model=model,
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        train_targets=train_targets,
        val_targets=val_targets,
    )

    dump_model(
        model=model,
        path=MODEL_PATH,
        name="xgb_regressor_100%_2",
    )
    predict_and_submit(
        model=model,
        fname="xgb_submission_100%_data",
        test_inputs=test_inputs,
    )
