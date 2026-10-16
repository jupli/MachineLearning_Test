#!/usr/bin/env bash
# End-to-end run instructions for the freight-rate prediction solution.
set -euo pipefail

echo "Installing solution dependencies..."
python -m pip install -r solution-requirements.txt

echo "Training model..."
python train.py

echo "Generating predictions..."
python predict.py

echo "Running official scorer to validate outputs + draw December chart..."
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

echo "Building the solution report..."
python make_report.py
