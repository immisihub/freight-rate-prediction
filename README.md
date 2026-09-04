# Freight Rate Prediction

Predicts truckload freight rates using historical shipment data, built for the 
Spotter Machine Learning Engineer assessment.

## Problem

Predict `posted_rate` for truckload shipments using shipment features (pickup, 
delivery, distance, equipment type, weight, market signals). The model is trained 
on historical data (Jan–Oct 2025) and must generalize to future, unseen loads 
(Nov–Dec 2025).

## Approach

- **Data validation & cleaning:** checked row counts, duplicate IDs, and missing 
  values (`weight`, `market_index`); imputed missing values using training-set 
  medians to avoid leakage.
- **Train/validation split:** time-based split (Jan–Aug train, Sep–Oct validation) 
  since the task is a forecasting problem, not simple interpolation.
- **Feature engineering:** date parts (month, day-of-week, day-of-year), haversine 
  distance as a sanity check, categorical encoding for pickup/delivery/equipment.
- **Baseline model:** Linear Regression — MAE $432.64, R² 0.66
- **Final model:** LightGBM — MAE $123.82, R² 0.82, MAPE 5.31%
- Final model retrained on full Jan–Oct data before generating predictions.

## How to Run

1. Install dependencies:
```bash
   pip install -r requirements.txt
```
2. Open `Assessment.ipynb` in Jupyter or Google Colab.
3. Run all cells in order. This will:
   - Load and clean the data
   - Train and validate the baseline and LightGBM models
   - Generate `validation_predictions.csv` (12,000 loads)
   - Generate `december_chart_inputs.csv` (31-day fixed-lane predictions)
4. Validate output format and generate the chart:
```bash
   python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
```

## Files

| File | Description |
|---|---|
| `Assessment.ipynb` | Full pipeline: EDA, cleaning, feature engineering, training, evaluation, prediction |
| `requirements.txt` | Python dependencies |
| `score.py` | Provided scorer — validates output format and generates the December chart |
| `validation_predictions.csv` | Final predictions for the 12,000 validation loads |
| `december_chart_inputs.csv` | Completed December sensitivity predictions (fixed lane) |

## Key Findings

- Distance is by far the strongest predictor of rate (feature importance gain 
  ~163x higher than the next feature).
- Equipment type affects the spread of high-value outliers more than the median rate.
- Day-of-week and month have minimal effect on rate (~3% variation), which is why 
  the December fixed-lane chart shows an essentially flat predicted rate — a 
  reflection of the model's learned behavior, not a bug.
