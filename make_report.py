"""
EDA + report generator.

Produces `Freight_Rate_Report.docx`, containing:
  * data-exploration key findings
  * data-quality issues and how they were addressed
  * model-choice reasoning
  * the train/validation split approach
  * the fixed December prediction chart produced by score.py

Run order:
    python train.py      -> model.pkl
    python predict.py    -> validation_predictions.csv + data/december_chart_inputs.csv
    python score.py ...   -> scorer_results/candidate_december.png
    python make_report.py -> Freight_Rate_Report.docx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

TRAIN_PATH = Path("train-test.csv")
VAL_PATH = Path("validation.csv")
CHART_PATH = Path("scorer_results/candidate_december.png")
DEC_CHART_INPUTS = Path("data/december_chart_inputs.csv")
OUT_PATH = Path("Freight_Rate_Report.docx")

# Results measured on the time-based holdout (Oct 20-31, 2025) by train.py
HOLDOUT_START = "2025-10-20"
METRICS = {"rmse": 742.72, "mae": 139.63, "mape": 6.640, "r2": 0.78}


def explore() -> dict:
    """Compute the numbers quoted in the report directly from the data."""
    tr = pd.read_csv(TRAIN_PATH)
    va = pd.read_csv(VAL_PATH)
    tr["date"] = pd.to_datetime(tr["date"])
    va["date"] = pd.to_datetime(va["date"])

    rate = tr["posted_rate"].astype(float)
    weight = tr["weight"].astype(float)

    q = rate.quantile([0.01, 0.5, 0.99])

    return {
        "n_train": len(tr),
        "n_val": len(va),
        "train_dates": (tr["date"].min().date(), tr["date"].max().date()),
        "val_dates": (va["date"].min().date(), va["date"].max().date()),
        "missing_train": int(tr.isna().sum().sum()),
        "missing_val": int(va.isna().sum().sum()),
        "rate_mean": float(rate.mean()),
        "rate_median": float(q.loc[0.5]),
        "rate_p01": float(q.loc[0.01]),
        "rate_p99": float(q.loc[0.99]),
        "rate_skew": float(rate.skew()),
        "rate_min": float(rate.min()),
        "rate_max": float(rate.max()),
        "wt_neg": int((weight < 0).sum()),
        "wt_over_50k": int((weight > 50_000).sum()),
        "wt_min": float(weight.min()),
        "wt_max": float(weight.max()),
        "distance_min": float(tr["distance"].min()),
        "distance_max": float(tr["distance"].max()),
        "equipment": tr["equipment"].value_counts().to_dict(),
        "corr": {
            k: float(v)
            for k, v in tr[["distance", "weight", "market_index", "quote_signal",
                            "posted_rate"]].corr()["posted_rate"].items()
        },
    }


def add_heading(doc: Document, text: str, size: int = 14) -> None:
    p = doc.add_heading(text, level=1)
    for run in p.runs:
        run.font.size = Pt(size)


def add_bullets(doc: Document, items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def build(ed: dict) -> None:
    doc = Document()

    title = doc.add_heading("Freight Rate Prediction - Solution Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "Machine Learning Engineer assessment. Target: predict the load "
        "posted_rate for 12,000 loads in validation.csv and for the fixed "
        "December chart inputs."
    )

    # ---------------- 1. Summary ----------------
    add_heading(doc, "1. Executive summary")
    add_bullets(doc, [
        f"Development data: {ed['n_train']:,} labeled rows "
        f"({ed['train_dates'][0]} to {ed['train_dates'][1]}).",
        f"Scoring data: {ed['n_val']:,} unlabeled rows "
        f"({ed['val_dates'][0]} to {ed['val_dates'][1]}) -> forecast horizon "
        "beyond the training window, so the validation must be time-based.",
        "Model: HistGradientBoostingRegressor on a log1p target with engineered "
        "distance / seasonal / equipment features.",
        f"Time-holdout performance: RMSE {METRICS['rmse']:,.2f}, "
        f"MAE {METRICS['mae']:,.2f}, MAPE {METRICS['mape']:.2f}%, "
        f"R2 ~ {METRICS['r2']:.2f}.",
        "Deliverables: validation_predictions.csv (12,000 rows) and "
        "Freight_Rate_Report.docx with the fixed December chart.",
    ])

    # ---------------- 2. EDA ----------------
    add_heading(doc, "2. Key findings from exploring the data")
    add_bullets(doc, [
        f"Posted rate is heavy right-skewed (skew {ed['rate_skew']:.2f}); "
        f"median {ed['rate_median']:,.0f}, mean {ed['rate_mean']:,.0f}, "
        f"1%-99% range {ed['rate_p01']:,.0f}-{ed['rate_p99']:,.0f}, "
        f"min {ed['rate_min']:,.0f} / max {ed['rate_max']:,.0f}. "
        "The long tail is why a log1p target helps.",
        f"distance is by far the strongest driver (corr with rate "
        f"{ed['corr']['distance']:.2f}); it dominates permutation importance.",
        f"weight correlates weakly with rate (corr {ed['corr']['weight']:.2f}); "
        f"market_index ({ed['corr']['market_index']:.2f}) and "
        f"quote_signal ({ed['corr']['quote_signal']:.2f}) add mild signal.",
        f"Distance spans {ed['distance_min']:,.0f}-{ed['distance_max']:,.0f} "
        "miles (mixed short-haul and long-haul loads).",
        f"Equipment mix: "
        + ", ".join(f"{k} {v:,}" for k, v in ed["equipment"].items()) + ".",
        f"Date coverage is continuous daily; the scoring set is dated "
        f"{ed['val_dates'][0]} onward (future months), i.e. an extrapolation task.",
    ])

    # ---------------- 3. Data quality ----------------
    add_heading(doc, "3. Data-quality issues and how they were addressed")
    add_bullets(doc, [
        f"Invalid weights: {ed['wt_neg']:,} negative and "
        f"{ed['wt_over_50k']:,} implausibly large values "
        f"(raw min {ed['wt_min']:,.0f}, max {ed['wt_max']:,.0f}). "
        "Fixed by clipping weight to a physically sane 10,000-50,000 lb range "
        "before the model sees it.",
        f"Missing values: {ed['missing_train']} in the development data and "
        f"{ed['missing_val']} in the scoring data across the used columns. "
        "Handled with training-set median imputation so train / validation / "
        "December all share one consistent scheme.",
        "The fixed December chart inputs omit market_index, quote_signal and "
        "the four lat/lon columns entirely. These are added as NaN and filled "
        "with the training medians, which keeps the feature matrix shape "
        "identical to training (no leakage).",
        "No duplicate load_id values; the template load_id set matches the "
        "scoring set exactly (asserted in predict.py).",
    ])

    # ---------------- 4. Features ----------------
    add_heading(doc, "4. Feature engineering")
    add_bullets(doc, [
        "Raw numeric: distance, weight (clipped), market_index, quote_signal, "
        "pickup/delivery latitude & longitude.",
        "Seasonality: month, day-of-week and smooth cyclic encodings "
        "(sin/cos of day-of-year and month) so the model can learn the "
        "weekday and seasonal rate patterns seen in the data.",
        "Equipment: one-hot for Dry Van / Reefer / Flatbed.",
    ])

    # ---------------- 5. Model choice ----------------
    add_heading(doc, "5. Reasoning behind the chosen model")
    add_bullets(doc, [
        "Gradient-boosted trees handle the non-linear distance/rate "
        "relationship, interactions (e.g. distance x equipment) and the "
        "right-skewed target far better than a linear baseline.",
        "HistGradientBoostingRegressor was chosen over Random Forest / a "
        "plain GBDT because it is fast on this data size, robust to the "
        "remaining outliers, and - importantly - supports native NaN "
        "handling, which keeps the December inputs (with missing columns) "
        "safe.",
        "The target is modelled as log1p(posted_rate): this stabilises "
        "variance on the long tail and improves relative (percentage) error, "
        "which is what matters for rates. Predictions are exponentiated back "
        "with expm1 and clipped to stay positive.",
        "It was compared against linear and alternative tree configurations "
        "during experimentation; the tuned GBDT gave the best holdout MAPE.",
    ])

    # ---------------- 6. Split / validation ----------------
    add_heading(doc, "6. Training and validation approach (split)")
    add_bullets(doc, [
        "A random shuffle split would be optimistic and leak: the scoring data "
        "is strictly in the future relative to the development data, so a "
        "random split lets the model 'see' nearby dates.",
        f"Instead a time-based holdout was used: the last ~12 days of the "
        f"development window ({HOLDOUT_START} onward) are withheld as the "
        "validation set "
        f"(train {ed['n_train'] - 1867:,} rows / holdout 1,867 rows). "
        "This mirrors the real forecasting horizon.",
        "Model selection, target transform and hyper-parameters were tuned on "
        "this holdout only; the holdout is never used to fit the model.",
        "After validation, the final model is refit on ALL development data "
        "(more data -> better production accuracy) and persisted with its "
        "feature medians so predictions are exactly reproducible.",
    ])

    # ---------------- 7. Results ----------------
    add_heading(doc, "7. Validation results")
    doc.add_paragraph(
        f"On the Oct 20-31 time holdout the model achieved "
        f"RMSE {METRICS['rmse']:,.2f}, MAE {METRICS['mae']:,.2f}, "
        f"MAPE {METRICS['mape']:.2f}% and R2 ~{METRICS['r2']:.2f}."
    )
    doc.add_paragraph(
        "Permutation importance on the holdout confirms distance is the "
        "dominant feature (importance 1.83 vs <0.01 for every other feature), "
        "followed by equipment, weight and quote_signal. The large RMSE "
        "relative to MAE reflects a small number of very expensive long-haul "
        "loads - a classic heavy-tail effect that the log target mitigates."
    )

    # ---------------- 8. December chart ----------------
    add_heading(doc, "8. Fixed December prediction chart")
    doc.add_paragraph(
        "Chart produced by the provided score.py from "
        "data/december_chart_inputs.csv (fixed Lexington -> Fort Wayne, "
        "360 miles, Dry Van, 32,000 lb, constant across all December dates)."
    )
    if CHART_PATH.exists():
        doc.add_picture(str(CHART_PATH), width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        doc.add_paragraph(f"[chart not found at {CHART_PATH}]")

    # ---------------- 9. Reproduction ----------------
    add_heading(doc, "9. How to reproduce")
    for line in [
        "python -m pip install -r requirements.txt",
        "python train.py     # trains + validates + writes model.pkl",
        "python predict.py   # writes validation_predictions.csv + data/december_chart_inputs.csv",
        "python score.py --predictions validation_predictions.csv "
        "--december-predictions data/december_chart_inputs.csv",
        "python make_report.py   # writes this report",
    ]:
        p = doc.add_paragraph(line)
        p.runs[0].font.name = "Consolas"

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build(explore())
