"""
Train the freight-rate model.

Steps
-----
1. Load data/train_test.csv (labeled development data, Jan-Oct 2025).
2. Clean + engineer features via the shared :mod:`features` module.
3. Validate on a *time-based* holdout (Oct 20-31) to mimic the Nov-Dec
   forecasting horizon and avoid look-ahead / shuffle leakage.
4. Report RMSE, MAE and MAPE on the holdout.
5. Refit on the FULL development set and persist the model + feature
   medians so predict.py can reproduce predictions exactly.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from features import FEATURE_COLS, FeatureEngineer

TRAIN_PATH = Path("train-test.csv")
MODEL_PATH = Path("model.pkl")
HOLDOUT_START = "2025-10-20"          # last ~12 days of October as validation
TARGET_TRANSFORM = "log1p"            # heavy right-skewed target -> log transform


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded development data: {df.shape}")

    y = np.log1p(df["posted_rate"].astype(float).values)   # log1p target

    fe = FeatureEngineer().fit(df)
    X = fe.transform(df)

    # ---- time-based validation split (no shuffle: preserves temporal order) ----
    holdout_mask = df["date"].dt.strftime("%Y-%m-%d") >= HOLDOUT_START
    X_tr, y_tr = X[~holdout_mask.values], y[~holdout_mask.values]
    X_va, y_va = X[holdout_mask.values], y[holdout_mask.values]
    print(f"Time holdout ({HOLDOUT_START}+): train={len(X_tr)} val={len(X_va)}")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=600,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred = np.expm1(model.predict(X_va))
    actual = np.expm1(y_va)
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((pred - actual) / actual)) * 100
    print(f"Holdout -> RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  MAPE: {mape:.3f}%")

    # Permutation importance on the holdout (model has no built-in feature_importances_)
    pi = permutation_importance(
        model, X_va, y_va, n_repeats=5, random_state=42, n_jobs=-1
    )
    imp = (
        pd.Series(pi.importances_mean, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )

    print("Top feature importances (permutation importance on holdout):")
    print(imp.head(10).to_string())

    # ---- refit on ALL development data for the final production model ----
    model.fit(X, y)

    MODEL_PATH.write_bytes(pickle.dumps({"model": model, "fe": fe}))
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
