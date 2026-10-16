"""Quick experiment to validate the modeling approach and measure RMSE."""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("train-test.csv")
print("Raw shape:", df.shape)

# ---- Data quality findings ----
# weight: has negatives (min -47500) and nulls -> clip to reasonable range
# market_index: has nulls -> leave for native NaN handling
# lat/lon: reported distance is route distance, not haversine (already a column)
print("weight neg count:", (df["weight"] < 0).sum())
print("weight nulls:", df["weight"].isna().sum())
print("market_index nulls:", df["market_index"].isna().sum())


def clean_weight(x):
    return x.clip(lower=10_000, upper=50_000)


def make_features(df, equipment_cats=None, fit=True):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = clean_weight(df["weight"])
    # impute market_index with median later handled by NaN-tolerant model; fill with global median for safety
    # date features
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)
    # equipment one-hot
    if fit:
        equipment_cats = sorted(df["equipment"].unique())
    for eq in equipment_cats:
        df[f"eq_{eq}"] = (df["equipment"] == eq).astype(float)
    # rate per mile target leakage check
    num_cols = ["distance", "weight", "market_index", "quote_signal",
                "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                "month", "dayofweek", "sin_doy", "cos_doy", "dayofyear"]
    X = df[num_cols + [f"eq_{e}" for e in equipment_cats]]
    return X, equipment_cats


X, equipment_cats = make_features(df)
y = df["posted_rate"].astype(float)

# Time-based split: hold out last ~20 days of October
df["date"] = pd.to_datetime(df["date"])
split_date = "2025-10-20"
mask = df["date"].dt.strftime("%Y-%m-%d") >= split_date
train_mask = ~mask

Xtr, ytr = X[train_mask.values if hasattr(train_mask, 'values') else train_mask], y[train_mask.values if hasattr(train_mask,'values') else train_mask]

# Re-do with aligned index
idx = df["date"].dt.strftime("%Y-%m-%d") >= split_date
Xtr = X[~idx.values]
ytr = y[~idx.values]
Xva = X[idx.values]
yva = y[idx.values]
print("Train:", Xtr.shape, "Val:", Xva.shape)

model = HistGradientBoostingRegressor(
    max_iter=600, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=False, random_state=42,
)
model.fit(Xtr, ytr)
pred = model.predict(Xva)
rmse = np.sqrt(np.mean((pred - yva) ** 2))
mae = mean_absolute_error(yva, pred)
print(f"Holdout RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2_score(yva, pred):.4f}")
print(f"yva range: {yva.min():.1f} - {yva.max():.1f}, mean: {yva.mean():.1f}")
print(f"pred range: {pred.min():.1f} - {pred.max():.1f}, mean: {pred.mean():.1f}")

# Feature importances
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importances:")
    print(imp.head(15))
