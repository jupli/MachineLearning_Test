"""
Generate final predictions.

Loads the trained model produced by train.py and writes:
  * validation_predictions.csv  -- predicted_rate for every row in
    validation.csv (12,000 rows), aligned to the template load_id order.
  * data/december_chart_inputs.csv -- the December fixed-input file with the
    predicted_rate column filled (needed by score.py to draw the chart).

Run order:  python train.py  then  python predict.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import FeatureEngineer

MODEL_PATH = Path("model.pkl")
VALIDATION_PATH = Path("validation.csv")
TEMPLATE_PATH = Path("validation-predictions-template.csv")
VALIDATION_OUT = Path("validation_predictions.csv")
DATA_DIR = Path("data")
DECEMBER_TEMPLATE = Path("december-chart-inputs.csv")
DECEMBER_OUT = DATA_DIR / "december_chart_inputs.csv"


def main() -> None:
    bundle = pickle.loads(MODEL_PATH.read_bytes())
    model, fe = bundle["model"], bundle["fe"]

    # ---- validation predictions ----
    val = pd.read_csv(VALIDATION_PATH)
    Xv = fe.transform(val)
    preds = np.expm1(model.predict(Xv))
    preds = np.clip(preds, 1.0, None)   # rates must be positive

    # Align to the template's load_id ordering (required column order: load_id,predicted_rate)
    template = pd.read_csv(TEMPLATE_PATH)
    out = template[["load_id"]].copy()
    out["predicted_rate"] = preds
    # Make sure every template id is covered exactly once, in template order
    merged = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val["load_id"], "predicted_rate": preds}),
        on="load_id", how="left",
    )
    assert merged["predicted_rate"].notna().all(), "some load_id missing predictions"
    assert len(merged) == 12000, f"expected 12000 rows, got {len(merged)}"
    merged.to_csv(VALIDATION_OUT, index=False)
    print(f"Wrote {VALIDATION_OUT}: {merged.shape}")

    # ---- December fixed-input predictions ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dec = pd.read_csv(DECEMBER_TEMPLATE)
    Xd = fe.transform(dec)
    dec_preds = np.expm1(model.predict(Xd))
    dec_preds = np.clip(dec_preds, 1.0, None)
    dec_out = dec.copy()
    dec_out["predicted_rate"] = dec_preds
    # Enforce exact column order required by score.py
    dec_out = dec_out[
        ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    ]
    dec_out.to_csv(DECEMBER_OUT, index=False)
    # also write back to the in-place template file used by readme
    dec_out.to_csv(DECEMBER_TEMPLATE, index=False)
    print(f"Wrote {DECEMBER_OUT}: {dec_out.shape}")
    print("December predicted rates:")
    print(dec_out[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
