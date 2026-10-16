# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

---

# Solution

## Approach in one paragraph

The scoring loads are dated in the future relative to the development data, so
validation uses a **time-based holdout** (last ~12 days of October) instead of a
random split. The target `posted_rate` is modelled as `log1p(rate)` because it is
heavily right-skewed. Engineered features are distance, cleaned weight,
market_index, quote_signal, pickup/delivery lat/lon, cyclic day-of-year / month
seasonality and one-hot equipment. The model is a
`HistGradientBoostingRegressor` (native NaN handling keeps the December inputs,
which omit several columns, safe). After tuning on the holdout the model is
refit on all development data.

Holdout results: **RMSE 742.72, MAE 139.63, MAPE 6.64%, R2 ~0.78**.

## Dependencies

```bash
python -m pip install -r solution-requirements.txt
```

## Run instructions (end to end)

Windows: `run.bat` &nbsp;|&nbsp; macOS/Linux: `bash run.sh`

Or run the steps manually:

```bash
python train.py     # cleans, engineers features, validates on the time holdout, saves model.pkl
python predict.py   # writes validation_predictions.csv and data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python make_report.py   # builds Freight_Rate_Report.docx (includes the December chart)
```

## Files

| File | Purpose |
| --- | --- |
| `features.py` | Shared cleaning + feature engineering (used by train and predict) |
| `train.py` | Time-based split, training, holdout metrics, permutation importance |
| `predict.py` | Final predictions for `validation.csv` + December chart inputs |
| `make_report.py` | EDA + generates `Freight_Rate_Report.docx` |
| `validation_predictions.csv` | Submission file (`load_id,predicted_rate`, 12,000 rows) |
| `Freight_Rate_Report.docx` | Submission report with the fixed December chart |
| `LOOM_SCRIPT.md` | Talking points / script for the 2-3 minute Loom video |

