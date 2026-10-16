"""Pick final config: compare on Oct 20+ time holdout (log target)."""
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from features import FeatureEngineer

df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])
y = np.log1p(df["posted_rate"].astype(float).values)
fe = FeatureEngineer().fit(df)
X = fe.transform(df)
mask = df["date"].dt.strftime("%Y-%m-%d") >= "2025-10-20"
Xtr,ytr = X[~mask.values], y[~mask.values]
Xva,yva = X[mask.values], y[mask.values]
ya = np.expm1(yva)

configs = [
  dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False),
  dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False),
  dict(max_iter=1500, learning_rate=0.02, max_leaf_nodes=63, l2_regularization=1.0, early_stopping=False),
  dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.5, early_stopping=False),
  # log target w/ native NaN (no median fill) to see Dec compatibility impact:
]
for c in configs:
    m = HistGradientBoostingRegressor(random_state=42, **c)
    m.fit(Xtr,ytr)
    p = np.expm1(m.predict(Xva))
    rmse=np.sqrt(np.mean((p-ya)**2)); mae=mean_absolute_error(ya,p); mape=np.mean(np.abs((p-ya)/ya))*100; r2=r2_score(ya,p)
    print(f"{c['max_iter']}/{c['learning_rate']}/{c['max_leaf_nodes']}/l2{c['l2_regularization']}: RMSE {rmse:.1f} MAE {mae:.1f} MAPE {mape:.3f}% R2 {r2:.3f}")
