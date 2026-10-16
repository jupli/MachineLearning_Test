"""Experiment: log target, tuning, metrics."""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])
y_raw = df["posted_rate"].astype(float)
y = np.log1p(y_raw)

def clean_weight(x):
    return x.clip(lower=10_000, upper=50_000)

df["weight_c"] = clean_weight(df["weight"])
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["dayofyear"] = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/365.0)
df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/365.0)
for eq in ["Dry Van","Reefer","Flatbed"]:
    df[f"eq_{eq}"] = (df["equipment"]==eq).astype(float)
num_cols = ["distance","weight_c","market_index","quote_signal",
            "pickup_lat","pickup_lon","delivery_lat","delivery_lon",
            "month","dayofweek","sin_doy","cos_doy","dayofyear"]
feat = num_cols + [f"eq_{e}" for e in ["Dry Van","Reefer","Flatbed"]]
X = df[feat]
# impute NaNs with median (market_index)
X = X.fillna(df[num_cols].median())

idx = df["date"].dt.strftime("%Y-%m-%d") >= "2025-10-20"
Xtr, ytr = X[~idx.values], y[~idx.values]
Xva, yva = X[idx.values], y[idx.values]

def eval_cfg(kwargs):
    m = HistGradientBoostingRegressor(random_state=42, **kwargs)
    m.fit(Xtr, ytr)
    p = np.expm1(m.predict(Xva))
    ya = np.expm1(yva)
    rmse = np.sqrt(np.mean((p-ya)**2))
    mae = mean_absolute_error(ya, p)
    mape = np.mean(np.abs((p-ya)/ya))*100
    print(f"{kwargs} -> RMSE {rmse:.2f} MAE {mae:.2f} MAPE {mape:.3f}  predmax {p.max():.1f} actmax {ya.max():.1f}")

eval_cfg(dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0))
eval_cfg(dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0))
eval_cfg(dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0))
eval_cfg(dict(max_iter=1000, learning_rate=0.02, max_leaf_nodes=31, l2_regularization=2.0))
eval_cfg(dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=127, l2_regularization=1.0, min_samples_leaf=20))
