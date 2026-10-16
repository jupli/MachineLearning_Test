"""Test native NaN handling (no median fill) for December compatibility."""
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def load_clean(path, with_target=True):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if with_target and "posted_rate" in df.columns:
        return df
    return df

df = load_clean("train-test.csv")
y = np.log1p(df["posted_rate"].astype(float).values)
df["weight"] = df["weight"].clip(lower=10_000, upper=50_000)
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["dayofyear"] = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/365.0)
df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/365.0)
df["sin_m"] = np.sin(2*np.pi*df["month"]/12.0)
df["cos_m"] = np.cos(2*np.pi*df["month"]/12.0)
for eq in ["Dry Van","Reefer","Flatbed"]:
    df[f"eq_{eq}"] = (df["equipment"]==eq).astype(float)
num_cols = ["distance","weight","market_index","quote_signal",
            "pickup_lat","pickup_lon","delivery_lat","delivery_lon",
            "month","dayofweek","sin_doy","cos_doy","sin_m","cos_m"]
feat = num_cols + [f"eq_{e}" for e in ["Dry Van","Reefer","Flatbed"]]
X = df[feat]  # NaN left as-is for native handling

idx = df["date"].dt.strftime("%Y-%m-%d") >= "2025-10-20"
Xtr, ytr = X[~idx.values], y[~idx.values]
Xva, yva = X[idx.values], y[idx.values]
ya = np.expm1(yva)

m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
    max_leaf_nodes=31, l2_regularization=1.0, random_state=42,
    early_stopping=False)
m.fit(Xtr, ytr)
p = np.expm1(m.predict(Xva))
rmse = np.sqrt(np.mean((p-ya)**2))
mae = mean_absolute_error(ya, p)
mape = np.mean(np.abs((p-ya)/ya))*100
print(f"native NaN: RMSE {rmse:.2f} MAE {mae:.2f} MAPE {mape:.3f} predmax {p.max():.1f} actmax {ya.max():.1f} predmin {p.min():.1f}")
print("NaN count in Xtr market_index:", Xtr["market_index"].isna().sum())
print("Any col still has NaN in Dec-style? (validation has all cols so fine)")
