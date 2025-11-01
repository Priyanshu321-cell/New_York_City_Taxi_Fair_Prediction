from features import load_parquet
from config import PROCESSED_DATA_PATH, RAW_DATA_PATH
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from dataset import load_csv
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import time


class MeanRegressor:
    def fit(self, inputs, targets):
        self.mean = targets.mean()

    def predict(self, inputs):
        return np.full(inputs.shape[0], self.mean)


def predict_and_submit(model, fname, test_inputs):
    """model.fit(train_inputs, train_targets)
    train_preds = model.predict(train_inputs)
    val_preds = model.predict(val_inputs)
    print(
        f"train loss : {root_mean_squared_error(train_targets, train_preds)} || Val Loss : {root_mean_squared_error(val_targets, val_preds)}"
    )"""
    test_preds = model.predict(test_inputs)
    submit = load_csv(RAW_DATA_PATH, name="sample_submission")
    submit["fare_amount"] = test_preds
    submit.to_csv(f"../data/submit_csv/{fname}.csv", index=False)


def evaluate_model(model, train_inputs, val_inputs, train_targets, val_targets):
    train_preds = model.predict(train_inputs)
    train_rmse = root_mean_squared_error(train_targets, train_preds)
    val_preds = model.predict(val_inputs)
    val_rmse = root_mean_squared_error(val_targets, val_preds)
    return train_rmse, val_rmse, train_preds, val_preds


def test_params(ModelClass, **params):
    """trains model with given params and return training and val rmse"""
    model = ModelClass(**params).fit(train_inputs, train_targets)
    train_rmse = root_mean_squared_error(
        train_targets.squeeze(1), model.predict(train_inputs)
    )
    val_rmse = root_mean_squared_error(
        val_targets.squeeze(1), model.predict(val_inputs)
    )
    return train_rmse, val_rmse


def test_params_and_plot(ModelClass, param_name, param_values, **other_params):
    """Trains multiple models by varying the values of param_name accor to params_values"""
    train_errors, val_errors = [], []
    for value in param_values:
        params = dict(other_params)
        params[param_name] = value
        train_rmse, val_rmse = test_params(ModelClass, **params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)

    plt.figure(figsize=(10, 6))
    plt.title("Overfitting curve: " + param_name)
    plt.plot(param_values, train_errors, "b-o")
    plt.plot(param_values, val_errors, "r-o")
    plt.xlabel(param_name)
    plt.ylabel("RMSE")
    plt.legend(["Training", "Validation"])


if __name__ == "__main__":
    train_inputs = load_parquet(PROCESSED_DATA_PATH, name="train_inputs")
    train_targets = load_parquet(PROCESSED_DATA_PATH, name="train_targets")
    val_inputs = load_parquet(PROCESSED_DATA_PATH, name="val_inputs")
    val_targets = load_parquet(PROCESSED_DATA_PATH, name="val_targets")
    test_inputs = load_parquet(PROCESSED_DATA_PATH, name="test_inputs")

    """mean model"""
    mean_model = MeanRegressor()
    mean_model.fit(train_inputs, train_targets)
    train_preds = mean_model.predict(train_inputs)
    val_preds = mean_model.predict(val_inputs)
    print(
        f"train loss : {root_mean_squared_error(train_targets, train_preds)} || Val Loss : {root_mean_squared_error(val_targets, val_preds)}"
    )

    """Ridge regression model"""
    model1 = Ridge(random_state=42, alpha=0.01).fit(train_inputs, train_targets)
    evaluate_model(model1)
    predict_and_submit(model1, fname="Ridge_regression_features_engineering")

    """Random Forest"""
    model2 = RandomForestRegressor(
        random_state=42, n_jobs=-1, max_depth=10, n_estimators=100
    ).fit(train_inputs, train_targets.squeeze(1))
    evaluate_model(model2)
    predict_and_submit(model2, fname="random_forest_regression_2")

    """XGB Regressor"""
    model3 = XGBRegressor(
        max_depth=5,
        objective="reg:squarederror",
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    model3.fit(train_inputs, train_targets.squeeze(1))
    evaluate_model(model3)
    predict_and_submit(model3, fname="xgb_submission_2")

    """Hypertuning"""
    best_params = {
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
    }

    """ n estimators"""
    s = time.time()
    test_params_and_plot(
        XGBRegressor, "n_estimators", [x for x in range(100, 800, 100)], **best_params
    )
    t = time.time()
    print(f" Time : {t - s}")

    best_params["n_estimators"] = 200

    """max depth"""
    s = time.time()
    test_params_and_plot(
        XGBRegressor, "max_depth", [x for x in range(1, 8)], **best_params
    )
    t = time.time()
    print(f" Time : {t - s}")

    best_params["max_depth"] = 5

    """learning rate"""
    s = time.time()
    test_params_and_plot(
        XGBRegressor,
        "learning_rate",
        [x for x in np.arange(0.01, 0.3, 0.02)],
        **best_params,
    )
    t = time.time()
    print(f" Time : {t - s}")

    best_params["learning_rate"] = 0.25

    """subsample"""
    s = time.time()
    test_params_and_plot(
        XGBRegressor,
        "subsample",
        [x for x in np.arange(0.1, 1, 0.05)],
        **best_params,
    )
    t = time.time()
    print(f" Time : {t - s}")

    best_params["subsample"] = 0.6

    """colsample_bytree"""
    s = time.time()
    test_params_and_plot(
        XGBRegressor,
        "colsample_bytree",
        [x for x in np.arange(0.1, 1, 0.05)],
        **best_params,
    )
    t = time.time()
    print(f" Time : {t - s}")

    best_params["colsample_bytree"] = 0.6

    evaluate_model(
        model=XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            learning_rate=0.08,
            n_estimators=500,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
        ).fit(train_inputs, train_targets)
    )
    predict_and_submit(
        model=XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            learning_rate=0.08,
            n_estimators=500,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
        ).fit(train_inputs, train_targets),
        fname="xgb_submission_3%_data",
    )
