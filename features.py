"""
Feature engineering for freight rate prediction.

Shared between training and prediction so that both use the exact same
column layout / imputation strategy (the trained model is loaded by
predict.py and applied to the validation and December inputs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]

# Numeric columns the model consumes. These are deliberately a fixed, ordered
# list so the matrix shape is identical for train / validation / December.
NUMERIC_COLS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "dayofweek",
    "sin_doy",
    "cos_doy",
    "sin_month",
    "cos_month",
]
EQUIPMENT_COLS = [f"eq_{e}" for e in EQUIPMENT]
FEATURE_COLS = NUMERIC_COLS + EQUIPMENT_COLS
TARGET = "posted_rate"

# Reasonable bounds for heavy-vehicle weight (in pounds). The raw data contains
# clearly invalid negative and implausibly large values which we clip.
WEIGHT_MIN = 10_000.0
WEIGHT_MAX = 50_000.0


class FeatureEngineer:
    """Stateful feature builder.

    ``fit`` learns imputation medians from the training data; ``transform``
    applies the same cleaning + encoding to any dataframe so train and
    inference stay byte-for-byte consistent. Median imputation (rather than
    leaving NaN) is used because the scoring set (validation.csv) reports
    market_index/weight as complete, while the December chart inputs are
    missing market_index, quote_signal *and* the four lat/lon columns --
    giving them the training median is the only consistent, leakage-free
    default.
    """

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        df = self._clean(df)
        self.median_ = df[NUMERIC_COLS].median(numeric_only=True)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean(df)
        # December chart inputs intentionally omit some optional columns
        # (market_index, quote_signal, lat/lon). Add them as NaN so the
        # imputation step below fills them with the training median and the
        # feature matrix shape stays identical to training.
        for col in NUMERIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        X = df[FEATURE_COLS].copy()
        # Fill any remaining NaN with training medians.
        for col in NUMERIC_COLS:
            if X[col].isna().any():
                X[col] = X[col].fillna(self.median_[col])
        return X

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # --- data-quality fixes ---
        # weight: clip impossible negatives / extremes to a sane range
        df["weight"] = df["weight"].clip(lower=WEIGHT_MIN, upper=WEIGHT_MAX)
        # date-derived seasonality features
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df["dayofyear"] = df["date"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
        df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
        for eq in EQUIPMENT:
            df[f"eq_{eq}"] = (df["equipment"] == eq).astype(float)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
