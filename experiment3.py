"""Experiment: model rate_per_mile then multiply by distance."""
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])
# rate per mile target (the natural multiplicative signal)
rpm = df["posted_rate"]/df["distance"]
print("rpm stats: mean", rpm.mean(), "std", rpm.std(), "min", rpm.min(), "max", rpm.max())

def clean_weight(x): return x.clip(lower=10_000, upper=50_000)
df["weight_c"] = clean_weight(df["weight"])
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek
df["dayofyear"] = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"]/365.0)
df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"]/365.0)
df["sin_m"] = np.sin(2*np.pi*df["month"]/12.0)
df["cos_m"] = np.cos(2*np.pi*df["month"]/12.0)
for eq in ["Dry Van","Reefer","Flatbed"]:
    df[f"eq_{eq}"] = (df["equipment"]==eq).astype(float)
num_cols = ["distance","weight_c","market_index","quote_signal",
            "pickup_lat","pickup_lon","delivery_lat","delivery_lon",
            "month","dayofweek","dayofyear","sin_doy","cos_doy","sin_m","cos_m"]
feat = num_cols + [f"eq_{e}" for e in ["Dry Van","Reefer","Flatbed"]]
X = df[feat].copy()
# fill market_index NaN with median
med = df[num_cols].median()
X = X.fillna(med)

idx = df["date"].dt.strftime("%Y-%m-%d") >= "2025-10-20"
Xtr, Xva = X[~idx.values], X[idx.values]

def run(label, yt_train, yt_val):
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=42, early_stopping=False)
    m.fit(Xtr, yt_train)
    p = m.predict(Xva)
    ya = yt_val
    rmse = np.sqrt(np.mean((p-ya)**2))
    mae = mean_absolute_error(ya, p)
    print(f"{label}: RMSE {rmse:.2f} MAE {mae:.2f}")
    return m

# Approach A: model rate_per_mile, predict, multiply by distance
ya_rpm = rpm[idx.values]
m = run("log(rpm) model", np.log1p(rpm[~idx.values]), ya_rpm)
pred_rpm = np.expm1(m.predict(Xva))
pred_rate = pred_rpm * df.loc[idx.index if False else df.index[idx.values], "distance"].values
ya_rate = df.loc[idx.values if False else df.index, "posted_rate"].values[idx.values]
# redo cleanly
va_idx = df.index[idx.values]
ya_rate = df.loc[va_idx, "posted_rate"].values
pred_rate = pred_rpm * df.loc[va_idx, "distance"].values
rmse_r = np.sqrt(np.mean((pred_rate-ya_rate)**2))
mae_r = mean_absolute_error(ya_rate, pred_rate)
mape_r = np.mean(np.abs((pred_rate-ya_rate)/ya_rate))*100
print(f"rate A (rpm->rate): RMSE {rmse_r:.2f} MAE {mae_r:.2f} MAPE {mape_r:.3f} predmax {pred_rate.max():.1f} actmax {ya_rate.max():.1f}")
